import base64
import binascii
import datetime
import email.utils
import functools
import gzip
import hashlib
import hmac
import http.cookies
from inspect import isclass
from io import BytesIO
import mimetypes
import numbers
import os.path
import re
import socket
import sys
import threading
import time
import warnings
import tornado
import traceback
import types
import urllib.parse
from urllib.parse import urlencode
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import escape
from tornado import gen
from tornado.httpserver import HTTPServer
from tornado import httputil
from tornado import iostream
from tornado import locale
from tornado.log import access_log, app_log, gen_log
from tornado import template
from tornado.escape import utf8, _unicode
from tornado.routing import (
from tornado.util import ObjectDict, unicode_type, _websocket_mask
from typing import (
from types import TracebackType
import typing
class GZipContentEncoding(OutputTransform):
    """Applies the gzip content encoding to the response.

    See http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.11

    .. versionchanged:: 4.0
        Now compresses all mime types beginning with ``text/``, instead
        of just a whitelist. (the whitelist is still used for certain
        non-text mime types).
    """
    CONTENT_TYPES = set(['application/javascript', 'application/x-javascript', 'application/xml', 'application/atom+xml', 'application/json', 'application/xhtml+xml', 'image/svg+xml'])
    GZIP_LEVEL = 6
    MIN_LENGTH = 1024

    def __init__(self, request: httputil.HTTPServerRequest) -> None:
        self._gzipping = 'gzip' in request.headers.get('Accept-Encoding', '')

    def _compressible_type(self, ctype: str) -> bool:
        return ctype.startswith('text/') or ctype in self.CONTENT_TYPES

    def transform_first_chunk(self, status_code: int, headers: httputil.HTTPHeaders, chunk: bytes, finishing: bool) -> Tuple[int, httputil.HTTPHeaders, bytes]:
        if 'Vary' in headers:
            headers['Vary'] += ', Accept-Encoding'
        else:
            headers['Vary'] = 'Accept-Encoding'
        if self._gzipping:
            ctype = _unicode(headers.get('Content-Type', '')).split(';')[0]
            self._gzipping = self._compressible_type(ctype) and (not finishing or len(chunk) >= self.MIN_LENGTH) and ('Content-Encoding' not in headers)
        if self._gzipping:
            headers['Content-Encoding'] = 'gzip'
            self._gzip_value = BytesIO()
            self._gzip_file = gzip.GzipFile(mode='w', fileobj=self._gzip_value, compresslevel=self.GZIP_LEVEL)
            chunk = self.transform_chunk(chunk, finishing)
            if 'Content-Length' in headers:
                if finishing:
                    headers['Content-Length'] = str(len(chunk))
                else:
                    del headers['Content-Length']
        return (status_code, headers, chunk)

    def transform_chunk(self, chunk: bytes, finishing: bool) -> bytes:
        if self._gzipping:
            self._gzip_file.write(chunk)
            if finishing:
                self._gzip_file.close()
            else:
                self._gzip_file.flush()
            chunk = self._gzip_value.getvalue()
            self._gzip_value.truncate(0)
            self._gzip_value.seek(0)
        return chunk