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
class BrotliDecoder(object):

    def __init__(self):
        self._obj = brotli.Decompressor()
        if hasattr(self._obj, 'decompress'):
            self.decompress = self._obj.decompress
        else:
            self.decompress = self._obj.process

    def flush(self):
        if hasattr(self._obj, 'flush'):
            return self._obj.flush()
        return b''