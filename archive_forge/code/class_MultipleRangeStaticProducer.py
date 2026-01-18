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
class MultipleRangeStaticProducer(StaticProducer):
    """
    A L{StaticProducer} that writes several chunks of a file to the request.
    """

    def __init__(self, request, fileObject, rangeInfo):
        """
        Initialize the instance.

        @param request: See L{StaticProducer}.
        @param fileObject: See L{StaticProducer}.
        @param rangeInfo: A list of tuples C{[(boundary, offset, size)]}
            where:
             - C{boundary} will be written to the request first.
             - C{offset} the offset into the file of chunk to write.
             - C{size} the size of the chunk to write.
        """
        StaticProducer.__init__(self, request, fileObject)
        self.rangeInfo = rangeInfo

    def start(self):
        self.rangeIter = iter(self.rangeInfo)
        self._nextRange()
        self.request.registerProducer(self, 0)

    def _nextRange(self):
        self.partBoundary, partOffset, self._partSize = next(self.rangeIter)
        self._partBytesWritten = 0
        self.fileObject.seek(partOffset)

    def resumeProducing(self):
        if not self.request:
            return
        data = []
        dataLength = 0
        done = False
        while dataLength < self.bufferSize:
            if self.partBoundary:
                dataLength += len(self.partBoundary)
                data.append(self.partBoundary)
                self.partBoundary = None
            p = self.fileObject.read(min(self.bufferSize - dataLength, self._partSize - self._partBytesWritten))
            self._partBytesWritten += len(p)
            dataLength += len(p)
            data.append(p)
            if self.request and self._partBytesWritten == self._partSize:
                try:
                    self._nextRange()
                except StopIteration:
                    done = True
                    break
        self.request.write(b''.join(data))
        if done:
            self.request.unregisterProducer()
            self.request.finish()
            self.stopProducing()