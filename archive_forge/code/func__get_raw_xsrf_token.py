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
def _get_raw_xsrf_token(self) -> Tuple[Optional[int], bytes, float]:
    """Read or generate the xsrf token in its raw form.

        The raw_xsrf_token is a tuple containing:

        * version: the version of the cookie from which this token was read,
          or None if we generated a new token in this request.
        * token: the raw token data; random (non-ascii) bytes.
        * timestamp: the time this token was generated (will not be accurate
          for version 1 cookies)
        """
    if not hasattr(self, '_raw_xsrf_token'):
        cookie_name = self.settings.get('xsrf_cookie_name', '_xsrf')
        cookie = self.get_cookie(cookie_name)
        if cookie:
            version, token, timestamp = self._decode_xsrf_token(cookie)
        else:
            version, token, timestamp = (None, None, None)
        if token is None:
            version = None
            token = os.urandom(16)
            timestamp = time.time()
        assert token is not None
        assert timestamp is not None
        self._raw_xsrf_token = (version, token, timestamp)
    return self._raw_xsrf_token