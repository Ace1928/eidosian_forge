import asyncio
import zlib
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Union  # noqa
from multidict import CIMultiDict
from .abc import AbstractStreamWriter
from .base_protocol import BaseProtocol
from .compression_utils import ZLibCompressor
from .helpers import NO_EXTENSIONS
def _py_serialize_headers(status_line: str, headers: 'CIMultiDict[str]') -> bytes:
    headers_gen = (_safe_header(k) + ': ' + _safe_header(v) for k, v in headers.items())
    line = status_line + '\r\n' + '\r\n'.join(headers_gen) + '\r\n\r\n'
    return line.encode('utf-8')