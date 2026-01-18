from __future__ import annotations
import collections
import io
import json as _json
import logging
import re
import sys
import typing
import warnings
import zlib
from contextlib import contextmanager
from http.client import HTTPMessage as _HttplibHTTPMessage
from http.client import HTTPResponse as _HttplibHTTPResponse
from socket import timeout as SocketTimeout
from . import util
from ._base_connection import _TYPE_BODY
from ._collections import HTTPHeaderDict
from .connection import BaseSSLError, HTTPConnection, HTTPException
from .exceptions import (
from .util.response import is_fp_closed, is_response_to_head
from .util.retry import Retry
class BytesQueueBuffer:
    """Memory-efficient bytes buffer

    To return decoded data in read() and still follow the BufferedIOBase API, we need a
    buffer to always return the correct amount of bytes.

    This buffer should be filled using calls to put()

    Our maximum memory usage is determined by the sum of the size of:

     * self.buffer, which contains the full data
     * the largest chunk that we will copy in get()

    The worst case scenario is a single chunk, in which case we'll make a full copy of
    the data inside get().
    """

    def __init__(self) -> None:
        self.buffer: typing.Deque[bytes] = collections.deque()
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def put(self, data: bytes) -> None:
        self.buffer.append(data)
        self._size += len(data)

    def get(self, n: int) -> bytes:
        if n == 0:
            return b''
        elif not self.buffer:
            raise RuntimeError('buffer is empty')
        elif n < 0:
            raise ValueError('n should be > 0')
        fetched = 0
        ret = io.BytesIO()
        while fetched < n:
            remaining = n - fetched
            chunk = self.buffer.popleft()
            chunk_length = len(chunk)
            if remaining < chunk_length:
                left_chunk, right_chunk = (chunk[:remaining], chunk[remaining:])
                ret.write(left_chunk)
                self.buffer.appendleft(right_chunk)
                self._size -= remaining
                break
            else:
                ret.write(chunk)
                self._size -= chunk_length
            fetched += chunk_length
            if not self.buffer:
                break
        return ret.getvalue()

    def get_all(self) -> bytes:
        buffer = self.buffer
        if not buffer:
            assert self._size == 0
            return b''
        if len(buffer) == 1:
            result = buffer.pop()
        else:
            ret = io.BytesIO()
            ret.writelines((buffer.popleft() for _ in range(len(buffer))))
            result = ret.getvalue()
        self._size = 0
        return result