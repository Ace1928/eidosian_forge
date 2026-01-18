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
def _init_decoder(self):
    """
        Set-up the _decoder attribute if necessary.
        """
    content_encoding = self.headers.get('content-encoding', '').lower()
    if self._decoder is None:
        if content_encoding in self.CONTENT_DECODERS:
            self._decoder = _get_decoder(content_encoding)
        elif ',' in content_encoding:
            encodings = [e.strip() for e in content_encoding.split(',') if e.strip() in self.CONTENT_DECODERS]
            if len(encodings):
                self._decoder = _get_decoder(content_encoding)