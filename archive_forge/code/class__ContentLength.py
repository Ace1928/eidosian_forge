import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
class _ContentLength(_SingleValueHeader):
    """
    Content-Length, RFC 2616 section 14.13

    Unlike other headers, use the CGI variable instead.
    """
    version = '1.0'
    _environ_name = 'CONTENT_LENGTH'