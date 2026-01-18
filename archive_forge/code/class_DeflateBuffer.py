import abc
import asyncio
import re
import string
from contextlib import suppress
from enum import IntEnum
from typing import (
from multidict import CIMultiDict, CIMultiDictProxy, istr
from yarl import URL
from . import hdrs
from .base_protocol import BaseProtocol
from .compression_utils import HAS_BROTLI, BrotliDecompressor, ZLibDecompressor
from .helpers import (
from .http_exceptions import (
from .http_writer import HttpVersion, HttpVersion10
from .log import internal_logger
from .streams import EMPTY_PAYLOAD, StreamReader
from .typedefs import RawHeaders
class DeflateBuffer:
    """DeflateStream decompress stream and feed data into specified stream."""
    decompressor: Any

    def __init__(self, out: StreamReader, encoding: Optional[str]) -> None:
        self.out = out
        self.size = 0
        self.encoding = encoding
        self._started_decoding = False
        self.decompressor: Union[BrotliDecompressor, ZLibDecompressor]
        if encoding == 'br':
            if not HAS_BROTLI:
                raise ContentEncodingError('Can not decode content-encoding: brotli (br). Please install `Brotli`')
            self.decompressor = BrotliDecompressor()
        else:
            self.decompressor = ZLibDecompressor(encoding=encoding)

    def set_exception(self, exc: BaseException) -> None:
        self.out.set_exception(exc)

    def feed_data(self, chunk: bytes, size: int) -> None:
        if not size:
            return
        self.size += size
        if not self._started_decoding and self.encoding == 'deflate' and (chunk[0] & 15 != 8):
            self.decompressor = ZLibDecompressor(encoding=self.encoding, suppress_deflate_header=True)
        try:
            chunk = self.decompressor.decompress_sync(chunk)
        except Exception:
            raise ContentEncodingError('Can not decode content-encoding: %s' % self.encoding)
        self._started_decoding = True
        if chunk:
            self.out.feed_data(chunk, len(chunk))

    def feed_eof(self) -> None:
        chunk = self.decompressor.flush()
        if chunk or self.size > 0:
            self.out.feed_data(chunk, len(chunk))
            if self.encoding == 'deflate' and (not self.decompressor.eof):
                raise ContentEncodingError('deflate')
        self.out.feed_eof()

    def begin_http_chunk_receiving(self) -> None:
        self.out.begin_http_chunk_receiving()

    def end_http_chunk_receiving(self) -> None:
        self.out.end_http_chunk_receiving()