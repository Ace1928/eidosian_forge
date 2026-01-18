from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
class ReverseProxyResourceTests(TestCase):
    """
    Tests for L{ReverseProxyResource}.
    """

    def _testRender(self, uri, expectedURI):
        """
        Check that a request pointing at C{uri} produce a new proxy connection,
        with the path of this request pointing at C{expectedURI}.
        """
        root = Resource()
        reactor = MemoryReactor()
        resource = ReverseProxyResource('127.0.0.1', 1234, b'/path', reactor)
        root.putChild(b'index', resource)
        site = Site(root)
        transport = StringTransportWithDisconnection()
        channel = site.buildProtocol(None)
        channel.makeConnection(transport)
        self.addCleanup(channel.connectionLost, None)
        channel.dataReceived(b'GET ' + uri + b' HTTP/1.1\r\nAccept: text/html\r\n\r\n')
        [(host, port, factory, _timeout, _bind_addr)] = reactor.tcpClients
        self.assertEqual(host, '127.0.0.1')
        self.assertEqual(port, 1234)
        self.assertIsInstance(factory, ProxyClientFactory)
        self.assertEqual(factory.rest, expectedURI)
        self.assertEqual(factory.headers[b'host'], b'127.0.0.1:1234')

    def test_render(self):
        """
        Test that L{ReverseProxyResource.render} initiates a connection to the
        given server with a L{ProxyClientFactory} as parameter.
        """
        return self._testRender(b'/index', b'/path')

    def test_render_subpage(self):
        """
        Test that L{ReverseProxyResource.render} will instantiate a child
        resource that will initiate a connection to the given server
        requesting the apropiate url subpath.
        """
        return self._testRender(b'/index/page1', b'/path/page1')

    def test_renderWithQuery(self):
        """
        Test that L{ReverseProxyResource.render} passes query parameters to the
        created factory.
        """
        return self._testRender(b'/index?foo=bar', b'/path?foo=bar')

    def test_getChild(self):
        """
        The L{ReverseProxyResource.getChild} method should return a resource
        instance with the same class as the originating resource, forward
        port, host, and reactor values, and update the path value with the
        value passed.
        """
        reactor = MemoryReactor()
        resource = ReverseProxyResource('127.0.0.1', 1234, b'/path', reactor)
        child = resource.getChild(b'foo', None)
        self.assertIsInstance(child, ReverseProxyResource)
        self.assertEqual(child.path, b'/path/foo')
        self.assertEqual(child.port, 1234)
        self.assertEqual(child.host, '127.0.0.1')
        self.assertIdentical(child.reactor, resource.reactor)

    def test_getChildWithSpecial(self):
        """
        The L{ReverseProxyResource} return by C{getChild} has a path which has
        already been quoted.
        """
        resource = ReverseProxyResource('127.0.0.1', 1234, b'/path')
        child = resource.getChild(b' /%', None)
        self.assertEqual(child.path, b'/path/%20%2F%25')