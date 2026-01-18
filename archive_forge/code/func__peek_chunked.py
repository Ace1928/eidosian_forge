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
def _peek_chunked(self, n):
    try:
        chunk_left = self._get_chunk_left()
    except IncompleteRead:
        return b''
    if chunk_left is None:
        return b''
    return self.fp.peek(chunk_left)[:chunk_left]