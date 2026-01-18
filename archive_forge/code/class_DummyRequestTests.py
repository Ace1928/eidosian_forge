from __future__ import annotations
from io import BytesIO
from typing import Dict, List, Optional
from zope.interface import implementer, verify
from incremental import Version
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, ISSLTransport
from twisted.internet.task import Clock
from twisted.python.deprecate import deprecated
from twisted.trial import unittest
from twisted.web._responses import FOUND
from twisted.web.http_headers import Headers
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Session, Site
class DummyRequestTests(unittest.SynchronousTestCase):
    """
    Tests for L{DummyRequest}.
    """

    def test_getClientIPDeprecated(self):
        """
        L{DummyRequest.getClientIP} is deprecated in favor of
        L{DummyRequest.getClientAddress}
        """
        request = DummyRequest([])
        request.getClientIP()
        warnings = self.flushWarnings(offendingFunctions=[self.test_getClientIPDeprecated])
        self.assertEqual(1, len(warnings))
        [warning] = warnings
        self.assertEqual(warning.get('category'), DeprecationWarning)
        self.assertEqual(warning.get('message'), 'twisted.web.test.requesthelper.DummyRequest.getClientIP was deprecated in Twisted 18.4.0; please use getClientAddress instead')

    def test_getClientIPSupportsIPv6(self):
        """
        L{DummyRequest.getClientIP} supports IPv6 addresses, just like
        L{twisted.web.http.Request.getClientIP}.
        """
        request = DummyRequest([])
        client = IPv6Address('TCP', '::1', 12345)
        request.client = client
        self.assertEqual('::1', request.getClientIP())

    def test_getClientAddressWithoutClient(self):
        """
        L{DummyRequest.getClientAddress} returns an L{IAddress}
        provider no C{client} has been set.
        """
        request = DummyRequest([])
        null = request.getClientAddress()
        verify.verifyObject(IAddress, null)

    def test_getClientAddress(self):
        """
        L{DummyRequest.getClientAddress} returns the C{client}.
        """
        request = DummyRequest([])
        client = IPv4Address('TCP', '127.0.0.1', 12345)
        request.client = client
        address = request.getClientAddress()
        self.assertIs(address, client)