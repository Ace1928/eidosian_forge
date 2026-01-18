import asyncio
import logging
import re
import types
from tornado.concurrent import (
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor
from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple
class _GzipMessageDelegate(httputil.HTTPMessageDelegate):
    """Wraps an `HTTPMessageDelegate` to decode ``Content-Encoding: gzip``."""

    def __init__(self, delegate: httputil.HTTPMessageDelegate, chunk_size: int) -> None:
        self._delegate = delegate
        self._chunk_size = chunk_size
        self._decompressor = None

    def headers_received(self, start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine], headers: httputil.HTTPHeaders) -> Optional[Awaitable[None]]:
        if headers.get('Content-Encoding', '').lower() == 'gzip':
            self._decompressor = GzipDecompressor()
            headers.add('X-Consumed-Content-Encoding', headers['Content-Encoding'])
            del headers['Content-Encoding']
        return self._delegate.headers_received(start_line, headers)

    async def data_received(self, chunk: bytes) -> None:
        if self._decompressor:
            compressed_data = chunk
            while compressed_data:
                decompressed = self._decompressor.decompress(compressed_data, self._chunk_size)
                if decompressed:
                    ret = self._delegate.data_received(decompressed)
                    if ret is not None:
                        await ret
                compressed_data = self._decompressor.unconsumed_tail
                if compressed_data and (not decompressed):
                    raise httputil.HTTPInputError('encountered unconsumed gzip data without making progress')
        else:
            ret = self._delegate.data_received(chunk)
            if ret is not None:
                await ret

    def finish(self) -> None:
        if self._decompressor is not None:
            tail = self._decompressor.flush()
            if tail:
                raise ValueError('decompressor.flush returned data; possible truncated input')
        return self._delegate.finish()

    def on_connection_close(self) -> None:
        return self._delegate.on_connection_close()