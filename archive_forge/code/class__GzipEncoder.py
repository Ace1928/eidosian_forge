import copy
import os
import re
import zlib
from binascii import hexlify
from html import escape
from typing import List, Optional
from urllib.parse import quote as _quote
from zope.interface import implementer
from incremental import Version
from twisted import copyright
from twisted.internet import address, interfaces
from twisted.internet.error import AlreadyCalled, AlreadyCancelled
from twisted.logger import Logger
from twisted.python import components, failure, reflect
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.spread.pb import Copyable, ViewPoint
from twisted.web import http, iweb, resource, util
from twisted.web.error import UnsupportedMethod
from twisted.web.http import unquote
@implementer(iweb._IRequestEncoder)
class _GzipEncoder:
    """
    An encoder which supports gzip.

    @ivar _zlibCompressor: The zlib compressor instance used to compress the
        stream.

    @ivar _request: A reference to the originating request.

    @since: 12.3
    """
    _zlibCompressor = None

    def __init__(self, compressLevel, request):
        self._zlibCompressor = zlib.compressobj(compressLevel, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        self._request = request

    def encode(self, data):
        """
        Write to the request, automatically compressing data on the fly.
        """
        if not self._request.startedWriting:
            self._request.responseHeaders.removeHeader(b'content-length')
        return self._zlibCompressor.compress(data)

    def finish(self):
        """
        Finish handling the request request, flushing any data from the zlib
        buffer.
        """
        remain = self._zlibCompressor.flush()
        self._zlibCompressor = None
        return remain