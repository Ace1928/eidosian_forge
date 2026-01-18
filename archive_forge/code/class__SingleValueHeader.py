import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
class _SingleValueHeader(HTTPHeader):
    """
    a ``HTTPHeader`` with exactly a single value

    This is the default behavior of ``HTTPHeader`` where returning a
    the string-value of headers via ``__call__`` assumes that only
    a single value exists.
    """
    pass