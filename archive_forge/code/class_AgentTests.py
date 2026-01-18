from __future__ import annotations
import zlib
from http.cookiejar import CookieJar
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple
from unittest import SkipTest, skipIf
from zope.interface.declarations import implementer
from zope.interface.verify import verifyObject
from incremental import Version
from twisted.internet import defer, task
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.endpoints import HostnameEndpoint, TCP4ClientEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import IOpenSSLClientConnectionCreator
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.test.test_endpoints import deterministicResolvingReactor
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import getDeprecationWarningString
from twisted.python.failure import Failure
from twisted.test.iosim import FakeTransport, IOPump
from twisted.test.test_sslverify import certificatesForAuthorityAndServer
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import client, error, http_headers
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.error import SchemeNotSupported
from twisted.web.http_headers import Headers
from twisted.web.iweb import (
from twisted.web.test.injectionhelpers import (
class AgentTests(TestCase, FakeReactorAndConnectMixin, AgentTestsMixin, IntegrationTestingMixin):
    """
    Tests for the new HTTP client API provided by L{Agent}.
    """

    def makeAgent(self):
        """
        @return: a new L{twisted.web.client.Agent} instance
        """
        return client.Agent(self.reactor)

    def setUp(self):
        """
        Create an L{Agent} wrapped around a fake reactor.
        """
        self.reactor = self.createReactor()
        self.agent = self.makeAgent()

    def test_defaultPool(self):
        """
        If no pool is passed in, the L{Agent} creates a non-persistent pool.
        """
        agent = client.Agent(self.reactor)
        self.assertIsInstance(agent._pool, HTTPConnectionPool)
        self.assertEqual(agent._pool.persistent, False)
        self.assertIdentical(agent._reactor, agent._pool._reactor)

    def test_persistent(self):
        """
        If C{persistent} is set to C{True} on the L{HTTPConnectionPool} (the
        default), C{Request}s are created with their C{persistent} flag set to
        C{True}.
        """
        pool = HTTPConnectionPool(self.reactor)
        agent = client.Agent(self.reactor, pool=pool)
        agent._getEndpoint = lambda *args: self
        agent.request(b'GET', b'http://127.0.0.1')
        self.assertEqual(self.protocol.requests[0][0].persistent, True)

    def test_nonPersistent(self):
        """
        If C{persistent} is set to C{False} when creating the
        L{HTTPConnectionPool}, C{Request}s are created with their
        C{persistent} flag set to C{False}.

        Elsewhere in the tests for the underlying HTTP code we ensure that
        this will result in the disconnection of the HTTP protocol once the
        request is done, so that the connection will not be returned to the
        pool.
        """
        pool = HTTPConnectionPool(self.reactor, persistent=False)
        agent = client.Agent(self.reactor, pool=pool)
        agent._getEndpoint = lambda *args: self
        agent.request(b'GET', b'http://127.0.0.1')
        self.assertEqual(self.protocol.requests[0][0].persistent, False)

    def test_connectUsesConnectionPool(self):
        """
        When a connection is made by the Agent, it uses its pool's
        C{getConnection} method to do so, with the endpoint returned by
        C{self._getEndpoint}. The key used is C{(scheme, host, port)}.
        """
        endpoint = DummyEndpoint()

        class MyAgent(client.Agent):

            def _getEndpoint(this, uri):
                self.assertEqual((uri.scheme, uri.host, uri.port), (b'http', b'foo', 80))
                return endpoint

        class DummyPool:
            connected = False
            persistent = False

            def getConnection(this, key, ep):
                this.connected = True
                self.assertEqual(ep, endpoint)
                self.assertEqual(key, (b'http', b'foo', 80))
                return defer.succeed(StubHTTPProtocol())
        pool = DummyPool()
        agent = MyAgent(self.reactor, pool=pool)
        self.assertIdentical(pool, agent._pool)
        headers = http_headers.Headers()
        headers.addRawHeader(b'host', b'foo')
        bodyProducer = object()
        agent.request(b'GET', b'http://foo/', bodyProducer=bodyProducer, headers=headers)
        self.assertEqual(agent._pool.connected, True)

    def test_nonBytesMethod(self):
        """
        L{Agent.request} raises L{TypeError} when the C{method} argument isn't
        L{bytes}.
        """
        self.assertRaises(TypeError, self.agent.request, 'GET', b'http://foo.example/')

    def test_unsupportedScheme(self):
        """
        L{Agent.request} returns a L{Deferred} which fails with
        L{SchemeNotSupported} if the scheme of the URI passed to it is not
        C{'http'}.
        """
        return self.assertFailure(self.agent.request(b'GET', b'mailto:alice@example.com'), SchemeNotSupported)

    def test_connectionFailed(self):
        """
        The L{Deferred} returned by L{Agent.request} fires with a L{Failure} if
        the TCP connection attempt fails.
        """
        result = self.agent.request(b'GET', b'http://foo/')
        host, port, factory = self.reactor.tcpClients.pop()[:3]
        factory.clientConnectionFailed(None, Failure(ConnectionRefusedError()))
        self.reactor.advance(10)
        self.failureResultOf(result, ConnectionRefusedError)

    def test_connectHTTP(self):
        """
        L{Agent._getEndpoint} return a C{HostnameEndpoint} when passed a scheme
        of C{'http'}.
        """
        expectedHost = b'example.com'
        expectedPort = 1234
        endpoint = self.agent._getEndpoint(URI.fromBytes(b'http://%b:%d' % (expectedHost, expectedPort)))
        self.assertEqual(endpoint._hostStr, 'example.com')
        self.assertEqual(endpoint._port, expectedPort)
        self.assertIsInstance(endpoint, HostnameEndpoint)

    def test_nonDecodableURI(self):
        """
        L{Agent._getEndpoint} when given a non-ASCII decodable URI will raise a
        L{ValueError} saying such.
        """
        uri = URI.fromBytes(b'http://example.com:80')
        uri.host = '☃.com'.encode()
        with self.assertRaises(ValueError) as e:
            self.agent._getEndpoint(uri)
        self.assertEqual(e.exception.args[0], 'The host of the provided URI ({reprout}) contains non-ASCII octets, it should be ASCII decodable.'.format(reprout=repr(uri.host)))

    def test_hostProvided(self):
        """
        If L{None} is passed to L{Agent.request} for the C{headers} parameter,
        a L{Headers} instance is created for the request and a I{Host} header
        added to it.
        """
        self.agent._getEndpoint = lambda *args: self
        self.agent.request(b'GET', b'http://example.com/foo?bar')
        req, res = self.protocol.requests.pop()
        self.assertEqual(req.headers.getRawHeaders(b'host'), [b'example.com'])

    def test_hostIPv6Bracketed(self):
        """
        If an IPv6 address is used in the C{uri} passed to L{Agent.request},
        the computed I{Host} header needs to be bracketed.
        """
        self.agent._getEndpoint = lambda *args: self
        self.agent.request(b'GET', b'http://[::1]/')
        req, res = self.protocol.requests.pop()
        self.assertEqual(req.headers.getRawHeaders(b'host'), [b'[::1]'])

    def test_hostOverride(self):
        """
        If the headers passed to L{Agent.request} includes a value for the
        I{Host} header, that value takes precedence over the one which would
        otherwise be automatically provided.
        """
        headers = http_headers.Headers({b'foo': [b'bar'], b'host': [b'quux']})
        self.agent._getEndpoint = lambda *args: self
        self.agent.request(b'GET', b'http://example.com/foo?bar', headers)
        req, res = self.protocol.requests.pop()
        self.assertEqual(req.headers.getRawHeaders(b'host'), [b'quux'])

    def test_headersUnmodified(self):
        """
        If a I{Host} header must be added to the request, the L{Headers}
        instance passed to L{Agent.request} is not modified.
        """
        headers = http_headers.Headers()
        self.agent._getEndpoint = lambda *args: self
        self.agent.request(b'GET', b'http://example.com/foo', headers)
        protocol = self.protocol
        self.assertEqual(len(protocol.requests), 1)
        self.assertEqual(headers, http_headers.Headers())

    def test_hostValueStandardHTTP(self):
        """
        When passed a scheme of C{'http'} and a port of C{80},
        L{Agent._computeHostValue} returns a string giving just
        the host name passed to it.
        """
        self.assertEqual(self.agent._computeHostValue(b'http', b'example.com', 80), b'example.com')

    def test_hostValueNonStandardHTTP(self):
        """
        When passed a scheme of C{'http'} and a port other than C{80},
        L{Agent._computeHostValue} returns a string giving the
        host passed to it joined together with the port number by C{":"}.
        """
        self.assertEqual(self.agent._computeHostValue(b'http', b'example.com', 54321), b'example.com:54321')

    def test_hostValueStandardHTTPS(self):
        """
        When passed a scheme of C{'https'} and a port of C{443},
        L{Agent._computeHostValue} returns a string giving just
        the host name passed to it.
        """
        self.assertEqual(self.agent._computeHostValue(b'https', b'example.com', 443), b'example.com')

    def test_hostValueNonStandardHTTPS(self):
        """
        When passed a scheme of C{'https'} and a port other than C{443},
        L{Agent._computeHostValue} returns a string giving the
        host passed to it joined together with the port number by C{":"}.
        """
        self.assertEqual(self.agent._computeHostValue(b'https', b'example.com', 54321), b'example.com:54321')

    def test_request(self):
        """
        L{Agent.request} establishes a new connection to the host indicated by
        the host part of the URI passed to it and issues a request using the
        method, the path portion of the URI, the headers, and the body producer
        passed to it.  It returns a L{Deferred} which fires with an
        L{IResponse} from the server.
        """
        self.agent._getEndpoint = lambda *args: self
        headers = http_headers.Headers({b'foo': [b'bar']})
        body = object()
        self.agent.request(b'GET', b'http://example.com:1234/foo?bar', headers, body)
        protocol = self.protocol
        self.assertEqual(len(protocol.requests), 1)
        req, res = protocol.requests.pop()
        self.assertIsInstance(req, Request)
        self.assertEqual(req.method, b'GET')
        self.assertEqual(req.uri, b'/foo?bar')
        self.assertEqual(req.headers, http_headers.Headers({b'foo': [b'bar'], b'host': [b'example.com:1234']}))
        self.assertIdentical(req.bodyProducer, body)

    def test_connectTimeout(self):
        """
        L{Agent} takes a C{connectTimeout} argument which is forwarded to the
        following C{connectTCP} agent.
        """
        agent = client.Agent(self.reactor, connectTimeout=5)
        agent.request(b'GET', b'http://foo/')
        timeout = self.reactor.tcpClients.pop()[3]
        self.assertEqual(5, timeout)

    @skipIf(not sslPresent, 'SSL not present, cannot run SSL tests.')
    def test_connectTimeoutHTTPS(self):
        """
        L{Agent} takes a C{connectTimeout} argument which is forwarded to the
        following C{connectTCP} call.
        """
        agent = client.Agent(self.reactor, connectTimeout=5)
        agent.request(b'GET', b'https://foo/')
        timeout = self.reactor.tcpClients.pop()[3]
        self.assertEqual(5, timeout)

    def test_bindAddress(self):
        """
        L{Agent} takes a C{bindAddress} argument which is forwarded to the
        following C{connectTCP} call.
        """
        agent = client.Agent(self.reactor, bindAddress='192.168.0.1')
        agent.request(b'GET', b'http://foo/')
        address = self.reactor.tcpClients.pop()[4]
        self.assertEqual('192.168.0.1', address)

    @skipIf(not sslPresent, 'SSL not present, cannot run SSL tests.')
    def test_bindAddressSSL(self):
        """
        L{Agent} takes a C{bindAddress} argument which is forwarded to the
        following C{connectSSL} call.
        """
        agent = client.Agent(self.reactor, bindAddress='192.168.0.1')
        agent.request(b'GET', b'https://foo/')
        address = self.reactor.tcpClients.pop()[4]
        self.assertEqual('192.168.0.1', address)

    def test_responseIncludesRequest(self):
        """
        L{Response}s returned by L{Agent.request} have a reference to the
        L{Request} that was originally issued.
        """
        uri = b'http://example.com/'
        agent = self.buildAgentForWrapperTest(self.reactor)
        d = agent.request(b'GET', uri)
        self.assertEqual(len(self.protocol.requests), 1)
        req, res = self.protocol.requests.pop()
        self.assertIsInstance(req, Request)
        resp = client.Response._construct((b'HTTP', 1, 1), 200, b'OK', Headers({}), None, req)
        res.callback(resp)
        response = self.successResultOf(d)
        self.assertEqual((response.request.method, response.request.absoluteURI, response.request.headers), (req.method, req.absoluteURI, req.headers))

    def test_requestAbsoluteURI(self):
        """
        L{Request.absoluteURI} is the absolute URI of the request.
        """
        uri = b'http://example.com/foo;1234?bar#frag'
        agent = self.buildAgentForWrapperTest(self.reactor)
        agent.request(b'GET', uri)
        self.assertEqual(len(self.protocol.requests), 1)
        req, res = self.protocol.requests.pop()
        self.assertIsInstance(req, Request)
        self.assertEqual(req.absoluteURI, uri)

    def test_requestMissingAbsoluteURI(self):
        """
        L{Request.absoluteURI} is L{None} if L{Request._parsedURI} is L{None}.
        """
        request = client.Request(b'FOO', b'/', Headers(), None)
        self.assertIdentical(request.absoluteURI, None)

    def test_endpointFactory(self):
        """
        L{Agent.usingEndpointFactory} creates an L{Agent} that uses the given
        factory to create endpoints.
        """
        factory = StubEndpointFactory()
        agent = client.Agent.usingEndpointFactory(None, endpointFactory=factory)
        uri = URI.fromBytes(b'http://example.com/')
        returnedEndpoint = agent._getEndpoint(uri)
        self.assertEqual(returnedEndpoint, (b'http', b'example.com', 80))

    def test_endpointFactoryDefaultPool(self):
        """
        If no pool is passed in to L{Agent.usingEndpointFactory}, a default
        pool is constructed with no persistent connections.
        """
        agent = client.Agent.usingEndpointFactory(self.reactor, StubEndpointFactory())
        pool = agent._pool
        self.assertEqual((pool.__class__, pool.persistent, pool._reactor), (HTTPConnectionPool, False, agent._reactor))

    def test_endpointFactoryPool(self):
        """
        If a pool is passed in to L{Agent.usingEndpointFactory} it is used as
        the L{Agent} pool.
        """
        pool = object()
        agent = client.Agent.usingEndpointFactory(self.reactor, StubEndpointFactory(), pool)
        self.assertIs(pool, agent._pool)