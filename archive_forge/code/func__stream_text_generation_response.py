import base64
import io
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
from requests import HTTPError
from ..constants import ENDPOINT
from ..utils import (
from ._text_generation import TextGenerationStreamResponse, _parse_text_generation_error
def _stream_text_generation_response(bytes_output_as_lines: Iterable[bytes], details: bool) -> Union[Iterable[str], Iterable[TextGenerationStreamResponse]]:
    for byte_payload in bytes_output_as_lines:
        if byte_payload == b'\n':
            continue
        payload = byte_payload.decode('utf-8')
        if payload.startswith('data:'):
            json_payload = json.loads(payload.lstrip('data:').rstrip('/n'))
            if json_payload.get('error') is not None:
                raise _parse_text_generation_error(json_payload['error'], json_payload.get('error_type'))
            output = TextGenerationStreamResponse(**json_payload)
            yield (output.token.text if not details else output)