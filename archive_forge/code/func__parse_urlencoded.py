from __future__ import annotations
import typing as t
from io import BytesIO
from urllib.parse import parse_qsl
from ._internal import _plain_int
from .datastructures import FileStorage
from .datastructures import Headers
from .datastructures import MultiDict
from .exceptions import RequestEntityTooLarge
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartDecoder
from .sansio.multipart import NeedData
from .wsgi import get_content_length
from .wsgi import get_input_stream
def _parse_urlencoded(self, stream: t.IO[bytes], mimetype: str, content_length: int | None, options: dict[str, str]) -> t_parse_result:
    if self.max_form_memory_size is not None and content_length is not None and (content_length > self.max_form_memory_size):
        raise RequestEntityTooLarge()
    try:
        items = parse_qsl(stream.read().decode(), keep_blank_values=True, errors='werkzeug.url_quote')
    except ValueError as e:
        raise RequestEntityTooLarge() from e
    return (stream, self.cls(items), self.cls())