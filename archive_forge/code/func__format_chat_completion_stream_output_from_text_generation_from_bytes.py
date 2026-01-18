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
def _format_chat_completion_stream_output_from_text_generation_from_bytes(byte_payload: bytes) -> Optional[ChatCompletionStreamOutput]:
    if not byte_payload.startswith(b'data:'):
        return None
    payload = byte_payload.decode('utf-8')
    json_payload = json.loads(payload.lstrip('data:').rstrip('/n'))
    return ChatCompletionStreamOutput.parse_obj_as_instance(json_payload)