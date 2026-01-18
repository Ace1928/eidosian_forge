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
def _init_length(self, request_method):
    """
        Set initial length value for Response content if available.
        """
    length = self.headers.get('content-length')
    if length is not None:
        if self.chunked:
            log.warning('Received response with both Content-Length and Transfer-Encoding set. This is expressly forbidden by RFC 7230 sec 3.3.2. Ignoring Content-Length and attempting to process response as Transfer-Encoding: chunked.')
            return None
        try:
            lengths = set([int(val) for val in length.split(',')])
            if len(lengths) > 1:
                raise InvalidHeader('Content-Length contained multiple unmatching values (%s)' % length)
            length = lengths.pop()
        except ValueError:
            length = None
        else:
            if length < 0:
                length = None
    try:
        status = int(self.status)
    except ValueError:
        status = 0
    if status in (204, 304) or 100 <= status < 200 or request_method == 'HEAD':
        length = 0
    return length