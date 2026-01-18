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
def _read1_chunked(self, n):
    chunk_left = self._get_chunk_left()
    if chunk_left is None or n == 0:
        return b''
    if not 0 <= n <= chunk_left:
        n = chunk_left
    read = self.fp.read1(n)
    self.chunk_left -= len(read)
    if not read:
        raise IncompleteRead(b'')
    return read