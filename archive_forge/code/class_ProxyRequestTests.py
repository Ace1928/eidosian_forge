from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
class ProxyRequestTests(TestCase):
    """
    Tests for L{ProxyRequest}.
    """

    def _testProcess(self, uri, expectedURI, method=b'GET', data=b''):
        """
        Build a request pointing at C{uri}, and check that a proxied request
        is created, pointing a C{expectedURI}.
        """
        transport = StringTransportWithDisconnection()
        channel = DummyChannel(transport)
        reactor = MemoryReactor()
        request = ProxyRequest(channel, False, reactor)
        request.gotLength(len(data))
        request.handleContentChunk(data)
        request.requestReceived(method, b'http://example.com' + uri, b'HTTP/1.0')
        self.assertEqual(len(reactor.tcpClients), 1)
        self.assertEqual(reactor.tcpClients[0][0], 'example.com')
        self.assertEqual(reactor.tcpClients[0][1], 80)
        factory = reactor.tcpClients[0][2]
        self.assertIsInstance(factory, ProxyClientFactory)
        self.assertEqual(factory.command, method)
        self.assertEqual(factory.version, b'HTTP/1.0')
        self.assertEqual(factory.headers, {b'host': b'example.com'})
        self.assertEqual(factory.data, data)
        self.assertEqual(factory.rest, expectedURI)
        self.assertEqual(factory.father, request)

    def test_process(self):
        """
        L{ProxyRequest.process} should create a connection to the given server,
        with a L{ProxyClientFactory} as connection factory, with the correct
        parameters:
            - forward comment, version and data values
            - update headers with the B{host} value
            - remove the host from the URL
            - pass the request as parent request
        """
        return self._testProcess(b'/foo/bar', b'/foo/bar')

    def test_processWithoutTrailingSlash(self):
        """
        If the incoming request doesn't contain a slash,
        L{ProxyRequest.process} should add one when instantiating
        L{ProxyClientFactory}.
        """
        return self._testProcess(b'', b'/')

    def test_processWithData(self):
        """
        L{ProxyRequest.process} should be able to retrieve request body and
        to forward it.
        """
        return self._testProcess(b'/foo/bar', b'/foo/bar', b'POST', b'Some content')

    def test_processWithPort(self):
        """
        Check that L{ProxyRequest.process} correctly parse port in the incoming
        URL, and create an outgoing connection with this port.
        """
        transport = StringTransportWithDisconnection()
        channel = DummyChannel(transport)
        reactor = MemoryReactor()
        request = ProxyRequest(channel, False, reactor)
        request.gotLength(0)
        request.requestReceived(b'GET', b'http://example.com:1234/foo/bar', b'HTTP/1.0')
        self.assertEqual(len(reactor.tcpClients), 1)
        self.assertEqual(reactor.tcpClients[0][0], 'example.com')
        self.assertEqual(reactor.tcpClients[0][1], 1234)