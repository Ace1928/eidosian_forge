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
Parse bytes from a Response object into a PIL Image.

    Expects the response body to be raw bytes. To deal with b64 encoded images, use `_b64_to_image` instead.
    