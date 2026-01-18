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
def _decode_content_transfer(self, data: bytes) -> bytes:
    encoding = self.headers.get(CONTENT_TRANSFER_ENCODING, '').lower()
    if encoding == 'base64':
        return base64.b64decode(data)
    elif encoding == 'quoted-printable':
        return binascii.a2b_qp(data)
    elif encoding in ('binary', '8bit', '7bit'):
        return data
    else:
        raise RuntimeError('unknown content transfer encoding: {}'.format(encoding))