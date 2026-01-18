from io import IOBase
from django.conf import settings
from django.core import signals
from django.core.handlers import base
from django.http import HttpRequest, QueryDict, parse_cookie
from django.urls import set_script_prefix
from django.utils.encoding import repercent_broken_unicode
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
class LimitedStream(IOBase):
    """
    Wrap another stream to disallow reading it past a number of bytes.

    Based on the implementation from werkzeug.wsgi.LimitedStream
    See https://github.com/pallets/werkzeug/blob/dbf78f67/src/werkzeug/wsgi.py#L828
    """

    def __init__(self, stream, limit):
        self._read = stream.read
        self._readline = stream.readline
        self._pos = 0
        self.limit = limit

    def read(self, size=-1, /):
        _pos = self._pos
        limit = self.limit
        if _pos >= limit:
            return b''
        if size == -1 or size is None:
            size = limit - _pos
        else:
            size = min(size, limit - _pos)
        data = self._read(size)
        self._pos += len(data)
        return data

    def readline(self, size=-1, /):
        _pos = self._pos
        limit = self.limit
        if _pos >= limit:
            return b''
        if size == -1 or size is None:
            size = limit - _pos
        else:
            size = min(size, limit - _pos)
        line = self._readline(size)
        self._pos += len(line)
        return line