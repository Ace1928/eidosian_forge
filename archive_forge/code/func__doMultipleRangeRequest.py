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
def _doMultipleRangeRequest(self, request, byteRanges):
    """
        Set up the response for Range headers that specify a single range.

        This method checks if the request is satisfiable and sets the response
        code and Content-Type and Content-Length headers appropriately.  The
        return value, which is a little complicated, indicates which parts of
        the resource to return and the boundaries that should separate the
        parts.

        In detail, the return value is a tuple rangeInfo C{rangeInfo} is a
        list of 3-tuples C{(partSeparator, partOffset, partSize)}.  The
        response to this request should be, for each element of C{rangeInfo},
        C{partSeparator} followed by C{partSize} bytes of the resource
        starting at C{partOffset}.  Each C{partSeparator} includes the
        MIME-style boundary and the part-specific Content-type and
        Content-range headers.  It is convenient to return the separator as a
        concrete string from this method, because this method needs to compute
        the number of bytes that will make up the response to be able to set
        the Content-Length header of the response accurately.

        @param request: The Request object.
        @param byteRanges: A list of C{(start, end)} values as specified by
            the header.  For each range, at most one of C{start} and C{end}
            may be L{None}.
        @return: See above.
        """
    matchingRangeFound = False
    rangeInfo = []
    contentLength = 0
    boundary = networkString(f'{int(time.time() * 1000000):x}{os.getpid():x}')
    if self.type:
        contentType = self.type
    else:
        contentType = b'bytes'
    for start, end in byteRanges:
        partOffset, partSize = self._rangeToOffsetAndSize(start, end)
        if partOffset == partSize == 0:
            continue
        contentLength += partSize
        matchingRangeFound = True
        partContentRange = self._contentRange(partOffset, partSize)
        partSeparator = networkString('\r\n--%s\r\nContent-type: %s\r\nContent-range: %s\r\n\r\n' % (nativeString(boundary), nativeString(contentType), nativeString(partContentRange)))
        contentLength += len(partSeparator)
        rangeInfo.append((partSeparator, partOffset, partSize))
    if not matchingRangeFound:
        request.setResponseCode(http.REQUESTED_RANGE_NOT_SATISFIABLE)
        request.setHeader(b'content-length', b'0')
        request.setHeader(b'content-range', networkString('bytes */%d' % (self.getFileSize(),)))
        return ([], b'')
    finalBoundary = b'\r\n--' + boundary + b'--\r\n'
    rangeInfo.append((finalBoundary, 0, 0))
    request.setResponseCode(http.PARTIAL_CONTENT)
    request.setHeader(b'content-type', networkString(f'multipart/byteranges; boundary="{nativeString(boundary)}"'))
    request.setHeader(b'content-length', b'%d' % (contentLength + len(finalBoundary),))
    return rangeInfo