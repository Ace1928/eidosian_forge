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
@implementer(interfaces.IPullProducer)
class StaticProducer:
    """
    Superclass for classes that implement the business of producing.

    @ivar request: The L{IRequest} to write the contents of the file to.
    @ivar fileObject: The file the contents of which to write to the request.
    """
    bufferSize = abstract.FileDescriptor.bufferSize

    def __init__(self, request, fileObject):
        """
        Initialize the instance.
        """
        self.request = request
        self.fileObject = fileObject

    def start(self):
        raise NotImplementedError(self.start)

    def resumeProducing(self):
        raise NotImplementedError(self.resumeProducing)

    def stopProducing(self):
        """
        Stop producing data.

        L{twisted.internet.interfaces.IProducer.stopProducing}
        is called when our consumer has died, and subclasses also call this
        method when they are done producing data.
        """
        self.fileObject.close()
        self.request = None