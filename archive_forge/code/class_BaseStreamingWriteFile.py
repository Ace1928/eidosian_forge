import hashlib
import io
import json
import os
import platform
import random
import socket
import ssl
import threading
import time
import urllib.parse
from typing import (
import filelock
import urllib3
from blobfile import _xml as xml
class BaseStreamingWriteFile(io.BufferedIOBase):

    def __init__(self, conf: Config, chunk_size: int) -> None:
        self._offset = 0
        self._buf = bytearray()
        self._chunk_size = chunk_size
        self._conf = conf

    def _upload_chunk(self, chunk: memoryview, finalize: bool) -> None:
        raise NotImplementedError

    def _upload_buf(self, buf: memoryview, finalize: bool=False) -> int:
        if finalize:
            size = len(buf)
        else:
            size = len(buf) // self._chunk_size * self._chunk_size
            assert size > 0
        chunk = buf[:size]
        self._upload_chunk(chunk, finalize)
        self._offset += len(chunk)
        return size

    def close(self) -> None:
        if self.closed:
            return
        size = self._upload_buf(memoryview(self._buf), finalize=True)
        assert size == len(self._buf)
        self._buf = bytearray()
        super().close()

    def tell(self) -> int:
        return self._offset + len(self._buf)

    def writable(self) -> bool:
        return True

    def write(self, b: bytes) -> int:
        if len(self._buf) == 0 and len(b) >= self._chunk_size:
            mv = memoryview(b)
            size = self._upload_buf(mv)
            self._buf = bytearray(mv[size:])
        else:
            self._buf += b
            if len(self._buf) >= self._chunk_size:
                mv = memoryview(self._buf)
                size = self._upload_buf(mv)
                self._buf = bytearray(mv[size:])
        assert len(self._buf) < self._chunk_size
        return len(b)

    def readinto(self, b: Any) -> int:
        raise io.UnsupportedOperation('not readable')

    def detach(self) -> io.RawIOBase:
        raise io.UnsupportedOperation('no underlying raw stream')

    def read1(self, size: int=-1) -> bytes:
        raise io.UnsupportedOperation('not readable')

    def readinto1(self, b: Any) -> int:
        raise io.UnsupportedOperation('not readable')