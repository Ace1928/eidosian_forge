from __future__ import annotations
import errno
import itertools
import mimetypes
import os
import time
import warnings
from html import escape
from typing import Any, Callable, Dict, Sequence
from urllib.parse import quote, unquote
from zope.interface import implementer
from incremental import Version
from typing_extensions import Literal
from twisted.internet import abstract, interfaces
from twisted.python import components, filepath, log
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecated
from twisted.python.runtime import platformType
from twisted.python.url import URL
from twisted.python.util import InsensitiveDict
from twisted.web import http, resource, server
from twisted.web.util import redirectTo
def _doSingleRangeRequest(self, request, startAndEnd):
    """
        Set up the response for Range headers that specify a single range.

        This method checks if the request is satisfiable and sets the response
        code and Content-Range header appropriately.  The return value
        indicates which part of the resource to return.

        @param request: The Request object.
        @param startAndEnd: A 2-tuple of start of the byte range as specified by
            the header and the end of the byte range as specified by the header.
            At most one of the start and end may be L{None}.
        @return: A 2-tuple of the offset and size of the range to return.
            offset == size == 0 indicates that the request is not satisfiable.
        """
    start, end = startAndEnd
    offset, size = self._rangeToOffsetAndSize(start, end)
    if offset == size == 0:
        request.setResponseCode(http.REQUESTED_RANGE_NOT_SATISFIABLE)
        request.setHeader(b'content-range', networkString('bytes */%d' % (self.getFileSize(),)))
    else:
        request.setResponseCode(http.PARTIAL_CONTENT)
        request.setHeader(b'content-range', self._contentRange(offset, size))
    return (offset, size)