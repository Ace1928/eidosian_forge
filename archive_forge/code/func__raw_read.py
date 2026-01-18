from __future__ import annotations
import collections
import io
import json as _json
import logging
import re
import sys
import typing
import warnings
import zlib
from contextlib import contextmanager
from http.client import HTTPMessage as _HttplibHTTPMessage
from http.client import HTTPResponse as _HttplibHTTPResponse
from socket import timeout as SocketTimeout
from . import util
from ._base_connection import _TYPE_BODY
from ._collections import HTTPHeaderDict
from .connection import BaseSSLError, HTTPConnection, HTTPException
from .exceptions import (
from .util.response import is_fp_closed, is_response_to_head
from .util.retry import Retry
def _raw_read(self, amt: int | None=None, *, read1: bool=False) -> bytes:
    """
        Reads `amt` of bytes from the socket.
        """
    if self._fp is None:
        return None
    fp_closed = getattr(self._fp, 'closed', False)
    with self._error_catcher():
        data = self._fp_read(amt, read1=read1) if not fp_closed else b''
        if amt is not None and amt != 0 and (not data):
            self._fp.close()
            if self.enforce_content_length and self.length_remaining is not None and (self.length_remaining != 0):
                raise IncompleteRead(self._fp_bytes_read, self.length_remaining)
        elif read1 and (amt != 0 and (not data) or self.length_remaining == len(data)):
            self._fp.close()
    if data:
        self._fp_bytes_read += len(data)
        if self.length_remaining is not None:
            self.length_remaining -= len(data)
    return data