import base64
import binascii
import json
import re
import uuid
import warnings
import zlib
from collections import deque
from types import TracebackType
from typing import (
from urllib.parse import parse_qsl, unquote, urlencode
from multidict import CIMultiDict, CIMultiDictProxy, MultiMapping
from .compression_utils import ZLibCompressor, ZLibDecompressor
from .hdrs import (
from .helpers import CHAR, TOKEN, parse_mimetype, reify
from .http import HeadersParser
from .payload import (
from .streams import StreamReader
@property
def _boundary_value(self) -> str:
    """Wrap boundary parameter value in quotes, if necessary.

        Reads self.boundary and returns a unicode string.
        """
    value = self._boundary
    if re.match(self._valid_tchar_regex, value):
        return value.decode('ascii')
    if re.search(self._invalid_qdtext_char_regex, value):
        raise ValueError('boundary value contains invalid characters')
    quoted_value_content = value.replace(b'\\', b'\\\\')
    quoted_value_content = quoted_value_content.replace(b'"', b'\\"')
    return '"' + quoted_value_content.decode('ascii') + '"'