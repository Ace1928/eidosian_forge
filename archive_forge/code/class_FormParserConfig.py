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
class FormParserConfig(TypedDict, total=False):
    UPLOAD_DIR: str | None
    UPLOAD_KEEP_FILENAME: bool
    UPLOAD_KEEP_EXTENSIONS: bool
    UPLOAD_ERROR_ON_BAD_CTE: bool
    MAX_MEMORY_FILE_SIZE: int
    MAX_BODY_SIZE: float