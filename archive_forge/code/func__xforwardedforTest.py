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
def _xforwardedforTest(self, header):
    """
        Assert that a request with the given value in its I{X-Forwarded-For}
        header is logged by L{proxiedLogFormatter} the same way it would have
        been logged by L{combinedLogFormatter} but with 172.16.1.2 as the
        client address instead of the normal value.

        @param header: An I{X-Forwarded-For} header with left-most address of
            172.16.1.2.
        """
    reactor = Clock()
    reactor.advance(1234567890)
    timestamp = http.datetimeToLogString(reactor.seconds())
    request = DummyRequestForLogTest(http.HTTPFactory(reactor=reactor))
    expected = http.combinedLogFormatter(timestamp, request).replace('1.2.3.4', '172.16.1.2')
    request.requestHeaders.setRawHeaders(b'x-forwarded-for', [header])
    line = http.proxiedLogFormatter(timestamp, request)
    self.assertEqual(expected, line)