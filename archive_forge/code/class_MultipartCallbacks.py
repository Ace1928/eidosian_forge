from __future__ import annotations
import logging
import os
import shutil
import sys
import tempfile
from email.message import Message
from enum import IntEnum
from io import BytesIO
from numbers import Number
from typing import TYPE_CHECKING
from .decoders import Base64Decoder, QuotedPrintableDecoder
from .exceptions import FileError, FormParserError, MultipartParseError, QuerystringParseError
class MultipartCallbacks(TypedDict, total=False):
    on_part_begin: Callable[[], None]
    on_part_data: Callable[[bytes, int, int], None]
    on_part_end: Callable[[], None]
    on_headers_begin: Callable[[], None]
    on_header_field: Callable[[bytes, int, int], None]
    on_header_value: Callable[[bytes, int, int], None]
    on_header_end: Callable[[], None]
    on_headers_finished: Callable[[], None]
    on_end: Callable[[], None]