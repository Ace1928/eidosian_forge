import calendar
import collections.abc
import copy
import datetime
import email.utils
from functools import lru_cache
from http.client import responses
import http.cookies
import re
from ssl import SSLError
import time
import unicodedata
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from tornado.escape import native_str, parse_qs_bytes, utf8
from tornado.log import gen_log
from tornado.util import ObjectDict, unicode_type
import typing
from typing import (
def _unquote_cookie(s: str) -> str:
    """Handle double quotes and escaping in cookie values.

    This method is copied verbatim from the Python 3.5 standard
    library (http.cookies._unquote) so we don't have to depend on
    non-public interfaces.
    """
    if s is None or len(s) < 2:
        return s
    if s[0] != '"' or s[-1] != '"':
        return s
    s = s[1:-1]
    i = 0
    n = len(s)
    res = []
    while 0 <= i < n:
        o_match = _OctalPatt.search(s, i)
        q_match = _QuotePatt.search(s, i)
        if not o_match and (not q_match):
            res.append(s[i:])
            break
        j = k = -1
        if o_match:
            j = o_match.start(0)
        if q_match:
            k = q_match.start(0)
        if q_match and (not o_match or k < j):
            res.append(s[i:k])
            res.append(s[k + 1])
            i = k + 2
        else:
            res.append(s[i:j])
            res.append(chr(int(s[j + 1:j + 4], 8)))
            i = j + 4
    return _nulljoin(res)