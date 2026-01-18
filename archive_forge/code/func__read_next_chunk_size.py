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
def _read_next_chunk_size(self):
    line = self.fp.readline(_MAXLINE + 1)
    if len(line) > _MAXLINE:
        raise LineTooLong('chunk size')
    i = line.find(b';')
    if i >= 0:
        line = line[:i]
    try:
        return int(line, 16)
    except ValueError:
        self._close_conn()
        raise