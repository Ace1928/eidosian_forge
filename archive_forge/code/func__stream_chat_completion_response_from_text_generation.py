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
def _stream_chat_completion_response_from_text_generation(text_generation_output: Iterable[TextGenerationStreamOutput]) -> Iterable[ChatCompletionStreamOutput]:
    """Used in `InferenceClient.chat_completion`."""
    created = int(time.time())
    for item in text_generation_output:
        yield _format_chat_completion_stream_output_from_text_generation(item, created)