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
def _setContentHeaders(self, request, size=None):
    """
        Set the Content-length and Content-type headers for this request.

        This method is not appropriate for requests for multiple byte ranges;
        L{_doMultipleRangeRequest} will set these headers in that case.

        @param request: The L{twisted.web.http.Request} object.
        @param size: The size of the response.  If not specified, default to
            C{self.getFileSize()}.
        """
    if size is None:
        size = self.getFileSize()
    request.setHeader(b'content-length', b'%d' % (size,))
    if self.type:
        request.setHeader(b'content-type', networkString(self.type))
    if self.encoding:
        request.setHeader(b'content-encoding', networkString(self.encoding))