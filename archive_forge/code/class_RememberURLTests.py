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
class RememberURLTests(unittest.TestCase):
    """
    Tests for L{server.Site}'s root request URL calculation.
    """

    def createServer(self, r):
        """
        Create a L{server.Site} bound to a L{DummyChannel} and the
        given resource as its root.

        @param r: The root resource.
        @type r: L{resource.Resource}

        @return: The channel to which the site is bound.
        @rtype: L{DummyChannel}
        """
        chan = DummyChannel()
        chan.site = server.Site(r)
        return chan

    def testSimple(self):
        """
        The path component of the root URL of a L{server.Site} whose
        root resource is below C{/} is that resource's path, and the
        netloc component is the L{site.Server}'s own host and port.
        """
        r = resource.Resource()
        r.isLeaf = 0
        rr = RootResource()
        r.putChild(b'foo', rr)
        rr.putChild(b'', rr)
        rr.putChild(b'bar', resource.Resource())
        chan = self.createServer(r)
        for url in [b'/foo/', b'/foo/bar', b'/foo/bar/baz', b'/foo/bar/']:
            request = server.Request(chan, 1)
            request.setHost(b'example.com', 81)
            request.gotLength(0)
            request.requestReceived(b'GET', url, b'HTTP/1.0')
            self.assertEqual(request.getRootURL(), b'http://example.com:81/foo')

    def testRoot(self):
        """
        The path component of the root URL of a L{server.Site} whose
        root resource is at C{/} is C{/}, and the netloc component is
        the L{site.Server}'s own host and port.
        """
        rr = RootResource()
        rr.putChild(b'', rr)
        rr.putChild(b'bar', resource.Resource())
        chan = self.createServer(rr)
        for url in [b'/', b'/bar', b'/bar/baz', b'/bar/']:
            request = server.Request(chan, 1)
            request.setHost(b'example.com', 81)
            request.gotLength(0)
            request.requestReceived(b'GET', url, b'HTTP/1.0')
            self.assertEqual(request.getRootURL(), b'http://example.com:81/')