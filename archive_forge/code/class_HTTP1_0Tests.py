import base64
import calendar
import random
from io import BytesIO
from itertools import cycle
from typing import Sequence, Union
from unittest import skipIf
from urllib.parse import clear_cache  # type: ignore[attr-defined]
from urllib.parse import urlparse, urlunsplit
from zope.interface import directlyProvides, providedBy, provider
from zope.interface.verify import verifyObject
import hamcrest
from twisted.internet import address
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols import loopback
from twisted.python.compat import iterbytes, networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
from twisted.web import http, http_headers, iweb
from twisted.web.http import PotentialDataLoss, _DataLoss, _IdentityTransferDecoder
from twisted.web.test.requesthelper import (
from ._util import assertIsFilesystemTemporary
class HTTP1_0Tests(unittest.TestCase, ResponseTestMixin):
    requests = b'GET / HTTP/1.0\r\n\r\nGET / HTTP/1.1\r\nAccept: text/html\r\n\r\n'
    expected_response: Union[Sequence[Sequence[bytes]], bytes] = [(b'HTTP/1.0 200 OK', b'Request: /', b'Command: GET', b'Version: HTTP/1.0', b'Content-Length: 13', b"'''\nNone\n'''\n")]

    def test_buffer(self):
        """
        Send requests over a channel and check responses match what is expected.
        """
        b = StringTransport()
        a = http.HTTPChannel()
        a.requestFactory = DummyHTTPHandlerProxy
        a.makeConnection(b)
        for byte in iterbytes(self.requests):
            a.dataReceived(byte)
        a.connectionLost(IOError('all one'))
        value = b.value()
        self.assertResponseEquals(value, self.expected_response)

    def test_requestBodyTimeout(self):
        """
        L{HTTPChannel} resets its timeout whenever data from a request body is
        delivered to it.
        """
        clock = Clock()
        transport = StringTransport()
        protocol = http.HTTPChannel()
        protocol.timeOut = 100
        protocol.callLater = clock.callLater
        protocol.makeConnection(transport)
        protocol.dataReceived(b'POST / HTTP/1.0\r\nContent-Length: 2\r\n\r\n')
        clock.advance(99)
        self.assertFalse(transport.disconnecting)
        protocol.dataReceived(b'x')
        clock.advance(99)
        self.assertFalse(transport.disconnecting)
        protocol.dataReceived(b'x')
        self.assertEqual(len(protocol.requests), 1)

    def test_requestBodyDefaultTimeout(self):
        """
        L{HTTPChannel}'s default timeout is 60 seconds.
        """
        clock = Clock()
        transport = StringTransport()
        factory = http.HTTPFactory()
        protocol = factory.buildProtocol(None)
        protocol = parametrizeTimeoutMixin(protocol, clock)
        protocol.makeConnection(transport)
        protocol.dataReceived(b'POST / HTTP/1.0\r\nContent-Length: 2\r\n\r\n')
        clock.advance(59)
        self.assertFalse(transport.disconnecting)
        clock.advance(1)
        self.assertTrue(transport.disconnecting)

    def test_transportForciblyClosed(self):
        """
        If a timed out transport doesn't close after 15 seconds, the
        L{HTTPChannel} will forcibly close it.
        """
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        clock = Clock()
        transport = StringTransport()
        factory = http.HTTPFactory()
        protocol = factory.buildProtocol(None)
        protocol = parametrizeTimeoutMixin(protocol, clock)
        protocol.makeConnection(transport)
        protocol.dataReceived(b'POST / HTTP/1.0\r\nContent-Length: 2\r\n\r\n')
        self.assertFalse(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        clock.advance(60)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        self.assertIn('Timing out client: {peer}', event['log_format'])
        clock.advance(14)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        clock.advance(1)
        self.assertTrue(transport.disconnecting)
        self.assertTrue(transport.disconnected)
        self.assertEquals(2, len(logObserver))
        event = logObserver[1]
        self.assertEquals('Forcibly timing out client: {peer}', event['log_format'])

    def test_transportNotAbortedAfterConnectionLost(self):
        """
        If a timed out transport ends up calling C{connectionLost}, it prevents
        the force-closure of the transport.
        """
        clock = Clock()
        transport = StringTransport()
        factory = http.HTTPFactory()
        protocol = factory.buildProtocol(None)
        protocol = parametrizeTimeoutMixin(protocol, clock)
        protocol.makeConnection(transport)
        protocol.dataReceived(b'POST / HTTP/1.0\r\nContent-Length: 2\r\n\r\n')
        self.assertFalse(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        clock.advance(60)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        clock.advance(14)
        protocol.connectionLost(None)
        clock.advance(1)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)

    def test_transportNotAbortedWithZeroAbortTimeout(self):
        """
        If the L{HTTPChannel} has its c{abortTimeout} set to L{None}, it never
        aborts.
        """
        clock = Clock()
        transport = StringTransport()
        factory = http.HTTPFactory()
        protocol = factory.buildProtocol(None)
        protocol._channel.abortTimeout = None
        protocol = parametrizeTimeoutMixin(protocol, clock)
        protocol.makeConnection(transport)
        protocol.dataReceived(b'POST / HTTP/1.0\r\nContent-Length: 2\r\n\r\n')
        self.assertFalse(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        clock.advance(60)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        clock.advance(2 ** 32)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)

    def test_connectionLostAfterForceClose(self):
        """
        If a timed out transport doesn't close after 15 seconds, the
        L{HTTPChannel} will forcibly close it.
        """
        clock = Clock()
        transport = StringTransport()
        factory = http.HTTPFactory()
        protocol = factory.buildProtocol(None)
        protocol = parametrizeTimeoutMixin(protocol, clock)
        protocol.makeConnection(transport)
        protocol.dataReceived(b'POST / HTTP/1.0\r\nContent-Length: 2\r\n\r\n')
        self.assertFalse(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        clock.advance(60)
        clock.advance(15)
        self.assertTrue(transport.disconnecting)
        self.assertTrue(transport.disconnected)
        protocol.connectionLost(ConnectionDone)

    def test_noPipeliningApi(self):
        """
        Test that a L{http.Request} subclass with no queued kwarg works as
        expected.
        """
        b = StringTransport()
        a = http.HTTPChannel()
        a.requestFactory = DummyHTTPHandlerProxy
        a.makeConnection(b)
        for byte in iterbytes(self.requests):
            a.dataReceived(byte)
        a.connectionLost(IOError('all done'))
        value = b.value()
        self.assertResponseEquals(value, self.expected_response)

    def test_noPipelining(self):
        """
        Test that pipelined requests get buffered, not processed in parallel.
        """
        b = StringTransport()
        a = http.HTTPChannel()
        a.requestFactory = DelayedHTTPHandlerProxy
        a.makeConnection(b)
        for byte in iterbytes(self.requests):
            a.dataReceived(byte)
        value = b.value()
        self.assertEqual(value, b'')
        self.assertEqual(1, len(a.requests))
        while a.requests:
            self.assertEqual(1, len(a.requests))
            request = a.requests[0].original
            request.delayedProcess()
        value = b.value()
        self.assertResponseEquals(value, self.expected_response)