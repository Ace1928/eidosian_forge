from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
class _HTTPParserTests:
    """
    Base test class for L{HTTPParser} which is responsible for the bulk of
    the task of parsing HTTP bytes.
    """
    sep: Optional[bytes] = None

    def test_statusCallback(self):
        """
        L{HTTPParser} calls its C{statusReceived} method when it receives a
        status line.
        """
        status = []
        protocol = HTTPParser()
        protocol.statusReceived = status.append
        protocol.makeConnection(StringTransport())
        self.assertEqual(protocol.state, STATUS)
        protocol.dataReceived(b'HTTP/1.1 200 OK' + self.sep)
        self.assertEqual(status, [b'HTTP/1.1 200 OK'])
        self.assertEqual(protocol.state, HEADER)

    def _headerTestSetup(self):
        header = {}
        protocol = HTTPParser()
        protocol.headerReceived = header.__setitem__
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK' + self.sep)
        return (header, protocol)

    def test_headerCallback(self):
        """
        L{HTTPParser} calls its C{headerReceived} method when it receives a
        header.
        """
        header, protocol = self._headerTestSetup()
        protocol.dataReceived(b'X-Foo:bar' + self.sep)
        protocol.dataReceived(self.sep)
        self.assertEqual(header, {b'X-Foo': b'bar'})
        self.assertEqual(protocol.state, BODY)

    def test_continuedHeaderCallback(self):
        """
        If a header is split over multiple lines, L{HTTPParser} calls
        C{headerReceived} with the entire value once it is received.
        """
        header, protocol = self._headerTestSetup()
        protocol.dataReceived(b'X-Foo: bar' + self.sep)
        protocol.dataReceived(b' baz' + self.sep)
        protocol.dataReceived(b'\tquux' + self.sep)
        protocol.dataReceived(self.sep)
        self.assertEqual(header, {b'X-Foo': b'bar baz\tquux'})
        self.assertEqual(protocol.state, BODY)

    def test_fieldContentWhitespace(self):
        """
        Leading and trailing linear whitespace is stripped from the header
        value passed to the C{headerReceived} callback.
        """
        header, protocol = self._headerTestSetup()
        value = self.sep.join([b' \t ', b' bar \t', b' \t', b''])
        protocol.dataReceived(b'X-Bar:' + value)
        protocol.dataReceived(b'X-Foo:' + value)
        protocol.dataReceived(self.sep)
        self.assertEqual(header, {b'X-Foo': b'bar', b'X-Bar': b'bar'})

    def test_allHeadersCallback(self):
        """
        After the last header is received, L{HTTPParser} calls
        C{allHeadersReceived}.
        """
        called = []
        header, protocol = self._headerTestSetup()

        def allHeadersReceived():
            called.append(protocol.state)
            protocol.state = STATUS
        protocol.allHeadersReceived = allHeadersReceived
        protocol.dataReceived(self.sep)
        self.assertEqual(called, [HEADER])
        self.assertEqual(protocol.state, STATUS)

    def test_noHeaderCallback(self):
        """
        If there are no headers in the message, L{HTTPParser} does not call
        C{headerReceived}.
        """
        header, protocol = self._headerTestSetup()
        protocol.dataReceived(self.sep)
        self.assertEqual(header, {})
        self.assertEqual(protocol.state, BODY)

    def test_headersSavedOnResponse(self):
        """
        All headers received by L{HTTPParser} are added to
        L{HTTPParser.headers}.
        """
        protocol = HTTPParser()
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK' + self.sep)
        protocol.dataReceived(b'X-Foo: bar' + self.sep)
        protocol.dataReceived(b'X-Foo: baz' + self.sep)
        protocol.dataReceived(self.sep)
        expected = [(b'X-Foo', [b'bar', b'baz'])]
        self.assertEqual(expected, list(protocol.headers.getAllRawHeaders()))

    def test_connectionControlHeaders(self):
        """
        L{HTTPParser.isConnectionControlHeader} returns C{True} for headers
        which are always connection control headers (similar to "hop-by-hop"
        headers from RFC 2616 section 13.5.1) and C{False} for other headers.
        """
        protocol = HTTPParser()
        connHeaderNames = [b'content-length', b'connection', b'keep-alive', b'te', b'trailers', b'transfer-encoding', b'upgrade', b'proxy-connection']
        for header in connHeaderNames:
            self.assertTrue(protocol.isConnectionControlHeader(header), "Expecting %r to be a connection control header, but wasn't" % (header,))
        self.assertFalse(protocol.isConnectionControlHeader(b'date'), "Expecting the arbitrarily selected 'date' header to not be a connection control header, but was.")

    def test_switchToBodyMode(self):
        """
        L{HTTPParser.switchToBodyMode} raises L{RuntimeError} if called more
        than once.
        """
        protocol = HTTPParser()
        protocol.makeConnection(StringTransport())
        protocol.switchToBodyMode(object())
        self.assertRaises(RuntimeError, protocol.switchToBodyMode, object())