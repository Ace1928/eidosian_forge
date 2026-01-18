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
def _parse_request_range(range_header: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """Parses a Range header.

    Returns either ``None`` or tuple ``(start, end)``.
    Note that while the HTTP headers use inclusive byte positions,
    this method returns indexes suitable for use in slices.

    >>> start, end = _parse_request_range("bytes=1-2")
    >>> start, end
    (1, 3)
    >>> [0, 1, 2, 3, 4][start:end]
    [1, 2]
    >>> _parse_request_range("bytes=6-")
    (6, None)
    >>> _parse_request_range("bytes=-6")
    (-6, None)
    >>> _parse_request_range("bytes=-0")
    (None, 0)
    >>> _parse_request_range("bytes=")
    (None, None)
    >>> _parse_request_range("foo=42")
    >>> _parse_request_range("bytes=1-2,6-10")

    Note: only supports one range (ex, ``bytes=1-2,6-10`` is not allowed).

    See [0] for the details of the range header.

    [0]: http://greenbytes.de/tech/webdav/draft-ietf-httpbis-p5-range-latest.html#byte.ranges
    """
    unit, _, value = range_header.partition('=')
    unit, value = (unit.strip(), value.strip())
    if unit != 'bytes':
        return None
    start_b, _, end_b = value.partition('-')
    try:
        start = _int_or_none(start_b)
        end = _int_or_none(end_b)
    except ValueError:
        return None
    if end is not None:
        if start is None:
            if end != 0:
                start = -end
                end = None
        else:
            end += 1
    return (start, end)