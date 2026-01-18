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
class MultipartPayloadWriter:

    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self._encoding: Optional[str] = None
        self._compress: Optional[ZLibCompressor] = None
        self._encoding_buffer: Optional[bytearray] = None

    def enable_encoding(self, encoding: str) -> None:
        if encoding == 'base64':
            self._encoding = encoding
            self._encoding_buffer = bytearray()
        elif encoding == 'quoted-printable':
            self._encoding = 'quoted-printable'

    def enable_compression(self, encoding: str='deflate', strategy: int=zlib.Z_DEFAULT_STRATEGY) -> None:
        self._compress = ZLibCompressor(encoding=encoding, suppress_deflate_header=True, strategy=strategy)

    async def write_eof(self) -> None:
        if self._compress is not None:
            chunk = self._compress.flush()
            if chunk:
                self._compress = None
                await self.write(chunk)
        if self._encoding == 'base64':
            if self._encoding_buffer:
                await self._writer.write(base64.b64encode(self._encoding_buffer))

    async def write(self, chunk: bytes) -> None:
        if self._compress is not None:
            if chunk:
                chunk = await self._compress.compress(chunk)
                if not chunk:
                    return
        if self._encoding == 'base64':
            buf = self._encoding_buffer
            assert buf is not None
            buf.extend(chunk)
            if buf:
                div, mod = divmod(len(buf), 3)
                enc_chunk, self._encoding_buffer = (buf[:div * 3], buf[div * 3:])
                if enc_chunk:
                    b64chunk = base64.b64encode(enc_chunk)
                    await self._writer.write(b64chunk)
        elif self._encoding == 'quoted-printable':
            await self._writer.write(binascii.b2a_qp(chunk))
        else:
            await self._writer.write(chunk)