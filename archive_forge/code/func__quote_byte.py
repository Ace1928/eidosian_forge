import base64
import codecs
import os
import posixpath
import re
import string
from typing import (
from urllib.parse import (
from urllib.parse import _coerce_args  # type: ignore
from urllib.request import pathname2url, url2pathname
from .util import to_unicode
from ._infra import _ASCII_TAB_OR_NEWLINE, _C0_CONTROL_OR_SPACE
from ._types import AnyUnicodeError, StrOrBytes
from ._url import _SPECIAL_SCHEMES
def _quote_byte(error: UnicodeError) -> Tuple[str, int]:
    error = cast(AnyUnicodeError, error)
    return (to_unicode(quote(error.object[error.start:error.end])), error.end)