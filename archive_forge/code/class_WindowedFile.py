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
class WindowedFile:
    """
    A file object that reads from a window into a file
    """

    def __init__(self, f: Any, start: int, end: int) -> None:
        self._f = f
        self._start = start
        self._end = end
        self._pos = -1
        self.seek(0)

    def tell(self) -> int:
        return self._pos - self._start

    def seek(self, offset: int, whence: int=io.SEEK_SET) -> None:
        new_pos = self._start + offset
        assert whence == io.SEEK_SET and self._start <= new_pos < self._end
        self._f.seek(new_pos, whence)
        self._pos = new_pos

    def read(self, n: Optional[int]=None) -> Any:
        assert self._pos <= self._end
        if n is None:
            n = self._end - self._pos
        n = min(n, self._end - self._pos)
        buf = self._f.read(n)
        self._pos += len(buf)
        if n > 0 and len(buf) == 0:
            raise Error('failed to read expected amount of data from file')
        assert self._pos <= self._end
        return buf