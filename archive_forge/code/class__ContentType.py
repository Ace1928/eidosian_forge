import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
class _ContentType(_SingleValueHeader):
    """
    Content-Type, RFC 2616 section 14.17

    Unlike other headers, use the CGI variable instead.
    """
    version = '1.0'
    _environ_name = 'CONTENT_TYPE'
    UNKNOWN = 'application/octet-stream'
    TEXT_PLAIN = 'text/plain'
    TEXT_HTML = 'text/html'
    TEXT_XML = 'text/xml'

    def compose(self, major=None, minor=None, charset=None):
        if not major:
            if minor in ('plain', 'html', 'xml'):
                major = 'text'
            else:
                assert not minor and (not charset)
                return (self.UNKNOWN,)
        if not minor:
            minor = '*'
        result = '%s/%s' % (major, minor)
        if charset:
            result += '; charset=%s' % charset
        return (result,)