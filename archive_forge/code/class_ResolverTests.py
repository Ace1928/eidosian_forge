import errno
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import defer
from twisted.internet.error import CannotListenError, ConnectionRefusedError
from twisted.internet.interfaces import IResolver
from twisted.internet.task import Clock
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.names import cache, client, dns, error, hosts
from twisted.names.common import ResolverBase
from twisted.names.error import DNSQueryTimeoutError
from twisted.names.test import test_util
from twisted.names.test.test_hosts import GoodTempPathMixin
from twisted.python import failure
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test import proto_helpers
from twisted.trial import unittest
class ResolverTests(unittest.TestCase):
    """
    Tests for L{client.Resolver}.
    """

    def test_clientProvidesIResolver(self):
        """
        L{client} provides L{IResolver} through a series of free
        functions.
        """
        verifyObject(IResolver, client)

    def test_clientResolverProvidesIResolver(self):
        """
        L{client.Resolver} provides L{IResolver}.
        """
        verifyClass(IResolver, client.Resolver)

    def test_noServers(self):
        """
        L{client.Resolver} raises L{ValueError} if constructed with neither
        servers nor a nameserver configuration file.
        """
        self.assertRaises(ValueError, client.Resolver)

    def test_missingConfiguration(self):
        """
        A missing nameserver configuration file results in no server information
        being loaded from it (ie, not an exception) and a default server being
        provided.
        """
        resolver = client.Resolver(resolv=self.mktemp(), reactor=Clock())
        self.assertEqual([('127.0.0.1', 53)], resolver.dynServers)

    def test_closesResolvConf(self):
        """
        As part of its constructor, C{StubResolver} opens C{/etc/resolv.conf};
        then, explicitly closes it and does not count on the GC to do so for
        it.
        """
        handle = FilePath(self.mktemp())
        resolvConf = handle.open(mode='w+')

        class StubResolver(client.Resolver):

            def _openFile(self, name):
                return resolvConf
        StubResolver(servers=['example.com', 53], resolv='/etc/resolv.conf', reactor=Clock())
        self.assertTrue(resolvConf.closed)

    def test_domainEmptyArgument(self):
        """
        L{client.Resolver.parseConfig} treats a I{domain} line without an
        argument as indicating a domain of C{b""}.
        """
        resolver = client.Resolver(servers=[('127.0.0.1', 53)])
        resolver.parseConfig([b'domain\n'])
        self.assertEqual(b'', resolver.domain)

    def test_searchEmptyArgument(self):
        """
        L{client.Resolver.parseConfig} treats a I{search} line without an
        argument as indicating an empty search suffix.
        """
        resolver = client.Resolver(servers=[('127.0.0.1', 53)])
        resolver.parseConfig([b'search\n'])
        self.assertEqual([], resolver.search)

    def test_datagramQueryServerOrder(self):
        """
        L{client.Resolver.queryUDP} should issue queries to its
        L{dns.DNSDatagramProtocol} with server addresses taken from its own
        C{servers} and C{dynServers} lists, proceeding through them in order
        as L{DNSQueryTimeoutError}s occur.
        """
        protocol = StubDNSDatagramProtocol()
        servers = [('::1', 53), ('::2', 53)]
        dynServers = [('::3', 53), ('::4', 53)]
        resolver = client.Resolver(servers=servers)
        resolver.dynServers = dynServers
        resolver._connectedProtocol = lambda interface: protocol
        expectedResult = object()
        queryResult = resolver.queryUDP(None)
        queryResult.addCallback(self.assertEqual, expectedResult)
        self.assertEqual(len(protocol.queries), 1)
        self.assertIs(protocol.queries[0][0], servers[0])
        protocol.queries[0][-1].errback(DNSQueryTimeoutError(0))
        self.assertEqual(len(protocol.queries), 2)
        self.assertIs(protocol.queries[1][0], servers[1])
        protocol.queries[1][-1].errback(DNSQueryTimeoutError(1))
        self.assertEqual(len(protocol.queries), 3)
        self.assertIs(protocol.queries[2][0], dynServers[0])
        protocol.queries[2][-1].errback(DNSQueryTimeoutError(2))
        self.assertEqual(len(protocol.queries), 4)
        self.assertIs(protocol.queries[3][0], dynServers[1])
        protocol.queries[3][-1].callback(expectedResult)
        return queryResult

    def test_singleConcurrentRequest(self):
        """
        L{client.Resolver.query} only issues one request at a time per query.
        Subsequent requests made before responses to prior ones are received
        are queued and given the same response as is given to the first one.
        """
        protocol = StubDNSDatagramProtocol()
        resolver = client.Resolver(servers=[('example.com', 53)])
        resolver._connectedProtocol = lambda: protocol
        queries = protocol.queries
        query = dns.Query(b'foo.example.com', dns.A, dns.IN)
        firstResult = resolver.query(query)
        self.assertEqual(len(queries), 1)
        secondResult = resolver.query(query)
        self.assertEqual(len(queries), 1)
        answer = object()
        response = dns.Message()
        response.answers.append(answer)
        queries.pop()[-1].callback(response)
        d = defer.gatherResults([firstResult, secondResult])

        def cbFinished(responses):
            firstResponse, secondResponse = responses
            self.assertEqual(firstResponse, ([answer], [], []))
            self.assertEqual(secondResponse, ([answer], [], []))
        d.addCallback(cbFinished)
        return d

    def test_multipleConcurrentRequests(self):
        """
        L{client.Resolver.query} issues a request for each different concurrent
        query.
        """
        protocol = StubDNSDatagramProtocol()
        resolver = client.Resolver(servers=[('example.com', 53)])
        resolver._connectedProtocol = lambda: protocol
        queries = protocol.queries
        firstQuery = dns.Query(b'foo.example.com', dns.A)
        resolver.query(firstQuery)
        self.assertEqual(len(queries), 1)
        secondQuery = dns.Query(b'bar.example.com', dns.A)
        resolver.query(secondQuery)
        self.assertEqual(len(queries), 2)
        thirdQuery = dns.Query(b'foo.example.com', dns.A6)
        resolver.query(thirdQuery)
        self.assertEqual(len(queries), 3)

    def test_multipleSequentialRequests(self):
        """
        After a response is received to a query issued with
        L{client.Resolver.query}, another query with the same parameters
        results in a new network request.
        """
        protocol = StubDNSDatagramProtocol()
        resolver = client.Resolver(servers=[('example.com', 53)])
        resolver._connectedProtocol = lambda: protocol
        queries = protocol.queries
        query = dns.Query(b'foo.example.com', dns.A)
        resolver.query(query)
        self.assertEqual(len(queries), 1)
        queries.pop()[-1].callback(dns.Message())
        resolver.query(query)
        self.assertEqual(len(queries), 1)

    def test_multipleConcurrentFailure(self):
        """
        If the result of a request is an error response, the Deferreds for all
        concurrently issued requests associated with that result fire with the
        L{Failure}.
        """
        protocol = StubDNSDatagramProtocol()
        resolver = client.Resolver(servers=[('example.com', 53)])
        resolver._connectedProtocol = lambda: protocol
        queries = protocol.queries
        query = dns.Query(b'foo.example.com', dns.A)
        firstResult = resolver.query(query)
        secondResult = resolver.query(query)

        class ExpectedException(Exception):
            pass
        queries.pop()[-1].errback(failure.Failure(ExpectedException()))
        return defer.gatherResults([self.assertFailure(firstResult, ExpectedException), self.assertFailure(secondResult, ExpectedException)])

    def test_connectedProtocol(self):
        """
        L{client.Resolver._connectedProtocol} returns a new
        L{DNSDatagramProtocol} connected to a new address with a
        cryptographically secure random port number.
        """
        resolver = client.Resolver(servers=[('example.com', 53)])
        firstProto = resolver._connectedProtocol()
        secondProto = resolver._connectedProtocol()
        self.assertIsNotNone(firstProto.transport)
        self.assertIsNotNone(secondProto.transport)
        self.assertNotEqual(firstProto.transport.getHost().port, secondProto.transport.getHost().port)
        return defer.gatherResults([defer.maybeDeferred(firstProto.transport.stopListening), defer.maybeDeferred(secondProto.transport.stopListening)])

    def test_resolverUsesOnlyParameterizedReactor(self):
        """
        If a reactor instance is supplied to L{client.Resolver}
        L{client.Resolver._connectedProtocol} should pass that reactor
        to L{twisted.names.dns.DNSDatagramProtocol}.
        """
        reactor = test_util.MemoryReactor()
        resolver = client.Resolver(resolv=self.mktemp(), reactor=reactor)
        proto = resolver._connectedProtocol()
        self.assertIs(proto._reactor, reactor)

    def test_differentProtocol(self):
        """
        L{client.Resolver._connectedProtocol} is called once each time a UDP
        request needs to be issued and the resulting protocol instance is used
        for that request.
        """
        resolver = client.Resolver(servers=[('example.com', 53)])
        protocols = []

        class FakeProtocol:

            def __init__(self):
                self.transport = StubPort()

            def query(self, address, query, timeout=10, id=None):
                protocols.append(self)
                return defer.succeed(dns.Message())
        resolver._connectedProtocol = FakeProtocol
        resolver.query(dns.Query(b'foo.example.com'))
        resolver.query(dns.Query(b'bar.example.com'))
        self.assertEqual(len(set(protocols)), 2)

    def test_ipv6Resolver(self):
        """
        If the resolver is ipv6, open a ipv6 port.
        """
        fake = test_util.MemoryReactor()
        resolver = client.Resolver(servers=[('::1', 53)], reactor=fake)
        resolver.query(dns.Query(b'foo.example.com'))
        [(proto, transport)] = fake.udpPorts.items()
        interface = transport.getHost().host
        self.assertEqual('::', interface)

    def test_disallowedPort(self):
        """
        If a port number is initially selected which cannot be bound, the
        L{CannotListenError} is handled and another port number is attempted.
        """
        ports = []

        class FakeReactor:

            def listenUDP(self, port, *args, **kwargs):
                ports.append(port)
                if len(ports) == 1:
                    raise CannotListenError(None, port, None)
        resolver = client.Resolver(servers=[('example.com', 53)])
        resolver._reactor = FakeReactor()
        resolver._connectedProtocol()
        self.assertEqual(len(set(ports)), 2)

    def test_disallowedPortRepeatedly(self):
        """
        If port numbers that cannot be bound are repeatedly selected,
        L{resolver._connectedProtocol} will give up eventually.
        """
        ports = []

        class FakeReactor:

            def listenUDP(self, port, *args, **kwargs):
                ports.append(port)
                raise CannotListenError(None, port, None)
        resolver = client.Resolver(servers=[('example.com', 53)])
        resolver._reactor = FakeReactor()
        self.assertRaises(CannotListenError, resolver._connectedProtocol)
        self.assertEqual(len(ports), 1000)

    def test_runOutOfFiles(self):
        """
        If the process is out of files, L{Resolver._connectedProtocol}
        will give up.
        """
        ports = []

        class FakeReactor:

            def listenUDP(self, port, *args, **kwargs):
                ports.append(port)
                err = OSError(errno.EMFILE, 'Out of files :(')
                raise CannotListenError(None, port, err)
        resolver = client.Resolver(servers=[('example.com', 53)])
        resolver._reactor = FakeReactor()
        exc = self.assertRaises(CannotListenError, resolver._connectedProtocol)
        self.assertEqual(exc.socketError.errno, errno.EMFILE)
        self.assertEqual(len(ports), 1)

    def test_differentProtocolAfterTimeout(self):
        """
        When a query issued by L{client.Resolver.query} times out, the retry
        uses a new protocol instance.
        """
        resolver = client.Resolver(servers=[('example.com', 53)])
        protocols = []
        results = [defer.fail(failure.Failure(DNSQueryTimeoutError(None))), defer.succeed(dns.Message())]

        class FakeProtocol:

            def __init__(self):
                self.transport = StubPort()

            def query(self, address, query, timeout=10, id=None):
                protocols.append(self)
                return results.pop(0)
        resolver._connectedProtocol = FakeProtocol
        resolver.query(dns.Query(b'foo.example.com'))
        self.assertEqual(len(set(protocols)), 2)

    def test_protocolShutDown(self):
        """
        After the L{Deferred} returned by L{DNSDatagramProtocol.query} is
        called back, the L{DNSDatagramProtocol} is disconnected from its
        transport.
        """
        resolver = client.Resolver(servers=[('example.com', 53)])
        protocols = []
        result = defer.Deferred()

        class FakeProtocol:

            def __init__(self):
                self.transport = StubPort()

            def query(self, address, query, timeout=10, id=None):
                protocols.append(self)
                return result
        resolver._connectedProtocol = FakeProtocol
        resolver.query(dns.Query(b'foo.example.com'))
        self.assertFalse(protocols[0].transport.disconnected)
        result.callback(dns.Message())
        self.assertTrue(protocols[0].transport.disconnected)

    def test_protocolShutDownAfterTimeout(self):
        """
        The L{DNSDatagramProtocol} created when an interim timeout occurs is
        also disconnected from its transport after the Deferred returned by its
        query method completes.
        """
        resolver = client.Resolver(servers=[('example.com', 53)])
        protocols = []
        result = defer.Deferred()
        results = [defer.fail(failure.Failure(DNSQueryTimeoutError(None))), result]

        class FakeProtocol:

            def __init__(self):
                self.transport = StubPort()

            def query(self, address, query, timeout=10, id=None):
                protocols.append(self)
                return results.pop(0)
        resolver._connectedProtocol = FakeProtocol
        resolver.query(dns.Query(b'foo.example.com'))
        self.assertFalse(protocols[1].transport.disconnected)
        result.callback(dns.Message())
        self.assertTrue(protocols[1].transport.disconnected)

    def test_protocolShutDownAfterFailure(self):
        """
        If the L{Deferred} returned by L{DNSDatagramProtocol.query} fires with
        a failure, the L{DNSDatagramProtocol} is still disconnected from its
        transport.
        """

        class ExpectedException(Exception):
            pass
        resolver = client.Resolver(servers=[('example.com', 53)])
        protocols = []
        result = defer.Deferred()

        class FakeProtocol:

            def __init__(self):
                self.transport = StubPort()

            def query(self, address, query, timeout=10, id=None):
                protocols.append(self)
                return result
        resolver._connectedProtocol = FakeProtocol
        queryResult = resolver.query(dns.Query(b'foo.example.com'))
        self.assertFalse(protocols[0].transport.disconnected)
        result.errback(failure.Failure(ExpectedException()))
        self.assertTrue(protocols[0].transport.disconnected)
        return self.assertFailure(queryResult, ExpectedException)

    def test_tcpDisconnectRemovesFromConnections(self):
        """
        When a TCP DNS protocol associated with a Resolver disconnects, it is
        removed from the Resolver's connection list.
        """
        resolver = client.Resolver(servers=[('example.com', 53)])
        protocol = resolver.factory.buildProtocol(None)
        protocol.makeConnection(None)
        self.assertIn(protocol, resolver.connections)
        protocol.connectionLost(None)
        self.assertNotIn(protocol, resolver.connections)

    def test_singleTCPQueryErrbackOnConnectionFailure(self):
        """
        The deferred returned by L{client.Resolver.queryTCP} will
        errback when the TCP connection attempt fails. The reason for
        the connection failure is passed as the argument to errback.
        """
        reactor = proto_helpers.MemoryReactor()
        resolver = client.Resolver(servers=[('192.0.2.100', 53)], reactor=reactor)
        d = resolver.queryTCP(dns.Query('example.com'))
        host, port, factory, timeout, bindAddress = reactor.tcpClients[0]

        class SentinelException(Exception):
            pass
        factory.clientConnectionFailed(reactor.connectors[0], failure.Failure(SentinelException()))
        self.failureResultOf(d, SentinelException)

    def test_multipleTCPQueryErrbackOnConnectionFailure(self):
        """
        All pending L{resolver.queryTCP} C{deferred}s will C{errback}
        with the same C{Failure} if the connection attempt fails.
        """
        reactor = proto_helpers.MemoryReactor()
        resolver = client.Resolver(servers=[('192.0.2.100', 53)], reactor=reactor)
        d1 = resolver.queryTCP(dns.Query('example.com'))
        d2 = resolver.queryTCP(dns.Query('example.net'))
        host, port, factory, timeout, bindAddress = reactor.tcpClients[0]

        class SentinelException(Exception):
            pass
        factory.clientConnectionFailed(reactor.connectors[0], failure.Failure(SentinelException()))
        f1 = self.failureResultOf(d1, SentinelException)
        f2 = self.failureResultOf(d2, SentinelException)
        self.assertIs(f1, f2)

    def test_reentrantTCPQueryErrbackOnConnectionFailure(self):
        """
        An errback on the deferred returned by
        L{client.Resolver.queryTCP} may trigger another TCP query.
        """
        reactor = proto_helpers.MemoryReactor()
        resolver = client.Resolver(servers=[('127.0.0.1', 10053)], reactor=reactor)
        q = dns.Query('example.com')
        d = resolver.queryTCP(q)

        def reissue(e):
            e.trap(ConnectionRefusedError)
            return resolver.queryTCP(q)
        d.addErrback(reissue)
        self.assertEqual(len(reactor.tcpClients), 1)
        self.assertEqual(len(reactor.connectors), 1)
        host, port, factory, timeout, bindAddress = reactor.tcpClients[0]
        f1 = failure.Failure(ConnectionRefusedError())
        factory.clientConnectionFailed(reactor.connectors[0], f1)
        self.assertEqual(len(reactor.tcpClients), 2)
        self.assertEqual(len(reactor.connectors), 2)
        self.assertNoResult(d)
        f2 = failure.Failure(ConnectionRefusedError())
        factory.clientConnectionFailed(reactor.connectors[1], f2)
        f = self.failureResultOf(d, ConnectionRefusedError)
        self.assertIs(f, f2)

    def test_pendingEmptiedInPlaceOnError(self):
        """
        When the TCP connection attempt fails, the
        L{client.Resolver.pending} list is emptied in place. It is not
        replaced with a new empty list.
        """
        reactor = proto_helpers.MemoryReactor()
        resolver = client.Resolver(servers=[('192.0.2.100', 53)], reactor=reactor)
        d = resolver.queryTCP(dns.Query('example.com'))
        host, port, factory, timeout, bindAddress = reactor.tcpClients[0]
        prePending = resolver.pending
        self.assertEqual(len(prePending), 1)

        class SentinelException(Exception):
            pass
        factory.clientConnectionFailed(reactor.connectors[0], failure.Failure(SentinelException()))
        self.failureResultOf(d, SentinelException)
        self.assertIs(resolver.pending, prePending)
        self.assertEqual(len(prePending), 0)