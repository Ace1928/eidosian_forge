from __future__ import absolute_import
import io
import logging
import zlib
from contextlib import contextmanager
from socket import error as SocketError
from socket import timeout as SocketTimeout
from ._collections import HTTPHeaderDict
from .connection import BaseSSLError, HTTPException
from .exceptions import (
from .packages import six
from .util.response import is_fp_closed, is_response_to_head
def _flush_decoder(self):
    """
        Flushes the decoder. Should only be called if the decoder is actually
        being used.
        """
    if self._decoder:
        buf = self._decoder.decompress(b'')
        return buf + self._decoder.flush()
    return b''