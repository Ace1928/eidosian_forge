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
def _import_pil_image():
    """Make sure `PIL` is installed on the machine."""
    if not is_pillow_available():
        raise ImportError("Please install Pillow to use deal with images (`pip install Pillow`). If you don't want the image to be post-processed, use `client.post(...)` and get the raw response from the server.")
    from PIL import Image
    return Image