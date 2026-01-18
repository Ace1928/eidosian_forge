import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
class _ContentRange(_SingleValueHeader):
    """
    Content-Range, RFC 2616 section 14.6
    """

    def compose(self, first_byte=None, last_byte=None, total_length=None):
        retval = 'bytes %d-%d/%d' % (first_byte, last_byte, total_length)
        assert last_byte == -1 or first_byte <= last_byte
        assert last_byte < total_length
        return (retval,)