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
class SingleRangeStaticProducer(StaticProducer):
    """
    A L{StaticProducer} that writes a single chunk of a file to the request.
    """

    def __init__(self, request, fileObject, offset, size):
        """
        Initialize the instance.

        @param request: See L{StaticProducer}.
        @param fileObject: See L{StaticProducer}.
        @param offset: The offset into the file of the chunk to be written.
        @param size: The size of the chunk to write.
        """
        StaticProducer.__init__(self, request, fileObject)
        self.offset = offset
        self.size = size

    def start(self):
        self.fileObject.seek(self.offset)
        self.bytesWritten = 0
        self.request.registerProducer(self, 0)

    def resumeProducing(self):
        if not self.request:
            return
        data = self.fileObject.read(min(self.bufferSize, self.size - self.bytesWritten))
        if data:
            self.bytesWritten += len(data)
            self.request.write(data)
        if self.request and self.bytesWritten == self.size:
            self.request.unregisterProducer()
            self.request.finish()
            self.stopProducing()