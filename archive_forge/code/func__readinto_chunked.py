import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
def _readinto_chunked(self, b):
    assert self.chunked != _UNKNOWN
    total_bytes = 0
    mvb = memoryview(b)
    try:
        while True:
            chunk_left = self._get_chunk_left()
            if chunk_left is None:
                return total_bytes
            if len(mvb) <= chunk_left:
                n = self._safe_readinto(mvb)
                self.chunk_left = chunk_left - n
                return total_bytes + n
            temp_mvb = mvb[:chunk_left]
            n = self._safe_readinto(temp_mvb)
            mvb = mvb[n:]
            total_bytes += n
            self.chunk_left = 0
    except IncompleteRead:
        raise IncompleteRead(bytes(b[0:total_bytes]))