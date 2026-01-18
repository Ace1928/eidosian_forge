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
def _first_or_none(items: List[Any]) -> Optional[Any]:
    try:
        return items[0] or None
    except IndexError:
        return None