from __future__ import (absolute_import, division,
from future.builtins import bytes, int, str, super
from future.utils import PY2
from future.backports.email import parser as email_parser
from future.backports.email import message as email_message
from future.backports.misc import create_connection as socket_create_connection
import io
import os
import socket
from future.backports.urllib.parse import urlsplit
import warnings
from array import array
def _readall_chunked(self):
    assert self.chunked != _UNKNOWN
    chunk_left = self.chunk_left
    value = []
    while True:
        if chunk_left is None:
            try:
                chunk_left = self._read_next_chunk_size()
                if chunk_left == 0:
                    break
            except ValueError:
                raise IncompleteRead(bytes(b'').join(value))
        value.append(self._safe_read(chunk_left))
        self._safe_read(2)
        chunk_left = None
    self._read_and_discard_trailer()
    self._close_conn()
    return bytes(b'').join(value)