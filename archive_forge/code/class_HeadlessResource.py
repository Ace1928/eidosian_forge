import os
import zlib
from io import BytesIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import interfaces
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.task import Clock
from twisted.internet.testing import EventLoggingObserver, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python import failure, reflect
from twisted.python.compat import iterbytes
from twisted.python.filepath import FilePath
from twisted.trial import unittest
from twisted.web import error, http, iweb, resource, server
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Request, Site
from twisted.web.static import Data
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from ._util import assertIsFilesystemTemporary
@implementer(resource.IResource)
class HeadlessResource:
    """
    A resource that implements GET but not HEAD.
    """
    allowedMethods = [b'GET']

    def render(self, request):
        """
        Leave the request open for future writes.
        """
        self.request = request
        if request.method not in self.allowedMethods:
            raise error.UnsupportedMethod(self.allowedMethods)
        self.request.write(b'some data')
        return server.NOT_DONE_YET

    def isLeaf(self):
        """
        # IResource.isLeaf
        """
        raise NotImplementedError()

    def getChildWithDefault(self, name, request):
        """
        # IResource.getChildWithDefault
        """
        raise NotImplementedError()

    def putChild(self, path, child):
        """
        # IResource.putChild
        """
        raise NotImplementedError()