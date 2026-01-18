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
@contextmanager
def _open_as_binary(content: Optional[ContentT]) -> Generator[Optional[BinaryT], None, None]:
    """Open `content` as a binary file, either from a URL, a local path, or raw bytes.

    Do nothing if `content` is None,

    TODO: handle a PIL.Image as input
    TODO: handle base64 as input
    """
    if isinstance(content, str):
        if content.startswith('https://') or content.startswith('http://'):
            logger.debug(f'Downloading content from {content}')
            yield get_session().get(content).content
            return
        content = Path(content)
        if not content.exists():
            raise FileNotFoundError(f'File not found at {content}. If `data` is a string, it must either be a URL or a path to a local file. To pass raw content, please encode it as bytes first.')
    if isinstance(content, Path):
        logger.debug(f'Opening content from {content}')
        with content.open('rb') as f:
            yield f
    else:
        yield content