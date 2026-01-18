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
def _parse_multipart(self, stream: t.IO[bytes], mimetype: str, content_length: int | None, options: dict[str, str]) -> t_parse_result:
    parser = MultiPartParser(stream_factory=self.stream_factory, max_form_memory_size=self.max_form_memory_size, max_form_parts=self.max_form_parts, cls=self.cls)
    boundary = options.get('boundary', '').encode('ascii')
    if not boundary:
        raise ValueError('Missing boundary')
    form, files = parser.parse(stream, boundary, content_length)
    return (stream, form, files)