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
class BaseStreamingReadFile(io.RawIOBase):

    def __init__(self, conf: Config, path: str, size: int) -> None:
        super().__init__()
        self._conf = conf
        self._size = size
        self._path = path
        self._offset = 0
        self._f = None
        self.requests = 0
        self.failures = 0
        self.bytes_read = 0

    def _request_chunk(self, streaming: bool, start: int, end: Optional[int]=None) -> 'urllib3.BaseHTTPResponse':
        raise NotImplementedError

    def readall(self) -> bytes:
        pieces = []
        while True:
            bytes_remaining = self._size - self._offset
            assert bytes_remaining >= 0, 'read more bytes than expected'
            opt_piece = self.read(min(CHUNK_SIZE, bytes_remaining))
            assert opt_piece is not None, 'file is in non-blocking mode'
            piece = opt_piece
            if len(piece) == 0:
                break
            pieces.append(piece)
        return b''.join(pieces)

    def readinto(self, b: Any) -> Optional[int]:
        bytes_remaining = self._size - self._offset
        if bytes_remaining <= 0 or len(b) == 0:
            return 0
        if not isinstance(b, memoryview):
            b = memoryview(b)
        if len(b) > bytes_remaining:
            b = b[:bytes_remaining]
        n = 0
        if self._conf.use_streaming_read:
            for attempt, backoff in enumerate(exponential_sleep_generator()):
                if self._f is None:
                    resp = self._request_chunk(streaming=True, start=self._offset)
                    if resp.status == 416:
                        return 0
                    self._f = resp
                    self.requests += 1
                err = None
                try:
                    opt_n = self._f.readinto(b)
                    assert opt_n is not None, 'file is in non-blocking mode'
                    n = opt_n
                    if n == 0:
                        err = Error(f'failed to read from connection while reading file at {self._path}')
                    else:
                        break
                except (urllib3.exceptions.ReadTimeoutError, urllib3.exceptions.ProtocolError, urllib3.exceptions.SSLError, ssl.SSLError) as e:
                    err = Error(f'exception {e} while reading file at {self._path}')
                self._f.close()
                self._f = None
                self.failures += 1
                if self._conf.retry_limit is not None and attempt >= self._conf.retry_limit:
                    raise err
                if attempt >= get_log_threshold_for_error(self._conf, str(err)):
                    self._conf.log_callback(f'error {err} when executing readinto({len(b)}) at offset {self._offset} attempt {attempt}, sleeping for {backoff:.1f} seconds before retrying')
                time.sleep(backoff)
        else:
            resp = self._request_chunk(streaming=False, start=self._offset, end=self._offset + len(b))
            if resp.status == 416:
                return 0
            self.requests += 1
            n = len(resp.data)
            b[:n] = resp.data
        self.bytes_read += n
        self._offset += n
        return n

    def seek(self, offset: int, whence: int=io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_offset = offset
        elif whence == io.SEEK_CUR:
            new_offset = self._offset + offset
        elif whence == io.SEEK_END:
            new_offset = self._size + offset
        else:
            raise ValueError(f'Invalid whence ({whence}, should be {io.SEEK_SET}, {io.SEEK_CUR}, or {io.SEEK_END})')
        if new_offset != self._offset:
            self._offset = new_offset
            if self._f is not None:
                self._f.close()
            self._f = None
        return self._offset

    def tell(self) -> int:
        return self._offset

    def close(self) -> None:
        if self.closed:
            return
        if hasattr(self, '_f') and self._f is not None:
            self._f.close()
            self._f = None
        super().close()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True