"""API Lambda que recibe {"texto": "..."} y devuelve respuesta de Groq."""

import base64
import binascii
import json
import os
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def json_response(status_code: int, payload: Any) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps(payload, ensure_ascii=False),
    }


def parse_request_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body")
    try:
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body, validate=True).decode("utf-8")
        payload = body if isinstance(body, dict) else json.loads(body)
    except (ValueError, TypeError, binascii.Error) as error:
        raise ValueError("El cuerpo debe ser un objeto JSON valido.") from error

    if not isinstance(payload, dict):
        raise ValueError("El cuerpo debe ser un objeto JSON valido.")
    return payload


def extract_texto(payload: Dict[str, Any]) -> str:
    texto = payload.get("texto")
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("Debes enviar un campo 'texto' con texto no vacio.")
    return texto.strip()


def call_groq(texto: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("La variable de entorno GROQ_API_KEY no esta configurada.")

    payload = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente que responde consultas. "
                    "Sigue las instrucciones del texto recibido."
                ),
            },
            {"role": "user", "content": texto},
        ],
        "temperature": 0.2,
    }
    request = Request(
        GROQ_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            )
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=20) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"Groq devolvio HTTP {error.code}.") from error
    except URLError as error:
        raise RuntimeError("No se pudo conectar con Groq.") from error
    except TimeoutError as error:
        raise RuntimeError("Se agoto el tiempo de espera de Groq.") from error
    except (ValueError, UnicodeError) as error:
        raise RuntimeError("Groq devolvio una respuesta invalida.") from error

    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("La respuesta de Groq no contiene contenido.") from error
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("La respuesta de Groq no contiene contenido.")
    return content


def lambda_handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    try:
        texto = extract_texto(parse_request_body(event))
    except ValueError as error:
        return json_response(400, {"error": str(error)})

    try:
        result = call_groq(texto)
    except RuntimeError as error:
        return json_response(
            502,
            {"error": "Fallo al invocar el API de Groq.", "detail": str(error)},
        )
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/plain; charset=utf-8"},
        "body": result,
    }
