import base64
import io
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
from requests import HTTPError
from huggingface_hub.errors import (
from ..constants import ENDPOINT
from ..utils import (
from ._generated.types import (
def _format_text_generation_stream_output(byte_payload: bytes, details: bool) -> Optional[Union[str, TextGenerationStreamOutput]]:
    if not byte_payload.startswith(b'data:'):
        return None
    payload = byte_payload.decode('utf-8')
    json_payload = json.loads(payload.lstrip('data:').rstrip('/n'))
    if json_payload.get('error') is not None:
        raise _parse_text_generation_error(json_payload['error'], json_payload.get('error_type'))
    output = TextGenerationStreamOutput.parse_obj_as_instance(json_payload)
    return output.token.text if not details else output