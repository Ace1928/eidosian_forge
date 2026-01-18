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
def _get_decoder(mode):
    if ',' in mode:
        return MultiDecoder(mode)
    if mode == 'gzip':
        return GzipDecoder()
    if brotli is not None and mode == 'br':
        return BrotliDecoder()
    return DeflateDecoder()