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
def _decode_xsrf_token(self, cookie: str) -> Tuple[Optional[int], Optional[bytes], Optional[float]]:
    """Convert a cookie string into a the tuple form returned by
        _get_raw_xsrf_token.
        """
    try:
        m = _signed_value_version_re.match(utf8(cookie))
        if m:
            version = int(m.group(1))
            if version == 2:
                _, mask_str, masked_token, timestamp_str = cookie.split('|')
                mask = binascii.a2b_hex(utf8(mask_str))
                token = _websocket_mask(mask, binascii.a2b_hex(utf8(masked_token)))
                timestamp = int(timestamp_str)
                return (version, token, timestamp)
            else:
                raise Exception('Unknown xsrf cookie version')
        else:
            version = 1
            try:
                token = binascii.a2b_hex(utf8(cookie))
            except (binascii.Error, TypeError):
                token = utf8(cookie)
            timestamp = int(time.time())
            return (version, token, timestamp)
    except Exception:
        gen_log.debug('Uncaught exception in _decode_xsrf_token', exc_info=True)
        return (None, None, None)