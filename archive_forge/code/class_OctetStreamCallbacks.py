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
class OctetStreamCallbacks(TypedDict, total=False):
    on_start: Callable[[], None]
    on_data: Callable[[bytes, int, int], None]
    on_end: Callable[[], None]