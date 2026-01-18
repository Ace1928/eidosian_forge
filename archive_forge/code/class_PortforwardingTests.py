from twisted.internet import address, defer, protocol, reactor
from twisted.protocols import portforward, wire
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class PortforwardingTests(unittest.TestCase):
    """
    Test port forwarding.
    """

    def setUp(self):
        self.serverProtocol = wire.Echo()
        self.clientProtocol = protocol.Protocol()
        self.openPorts = []

    def tearDown(self):
        try:
            self.proxyServerFactory.protoInstance.transport.loseConnection()
        except AttributeError:
            pass
        try:
            pi = self.proxyServerFactory.clientFactoryInstance.protoInstance
            pi.transport.loseConnection()
        except AttributeError:
            pass
        try:
            self.clientProtocol.transport.loseConnection()
        except AttributeError:
            pass
        try:
            self.serverProtocol.transport.loseConnection()
        except AttributeError:
            pass
        return defer.gatherResults([defer.maybeDeferred(p.stopListening) for p in self.openPorts])

    def test_portforward(self):
        """
        Test port forwarding through Echo protocol.
        """
        realServerFactory = protocol.ServerFactory()
        realServerFactory.protocol = lambda: self.serverProtocol
        realServerPort = reactor.listenTCP(0, realServerFactory, interface='127.0.0.1')
        self.openPorts.append(realServerPort)
        self.proxyServerFactory = TestableProxyFactory('127.0.0.1', realServerPort.getHost().port)
        proxyServerPort = reactor.listenTCP(0, self.proxyServerFactory, interface='127.0.0.1')
        self.openPorts.append(proxyServerPort)
        nBytes = 1000
        received = []
        d = defer.Deferred()

        def testDataReceived(data):
            received.extend(iterbytes(data))
            if len(received) >= nBytes:
                self.assertEqual(b''.join(received), b'x' * nBytes)
                d.callback(None)
        self.clientProtocol.dataReceived = testDataReceived

        def testConnectionMade():
            self.clientProtocol.transport.write(b'x' * nBytes)
        self.clientProtocol.connectionMade = testConnectionMade
        clientFactory = protocol.ClientFactory()
        clientFactory.protocol = lambda: self.clientProtocol
        reactor.connectTCP('127.0.0.1', proxyServerPort.getHost().port, clientFactory)
        return d

    def test_registerProducers(self):
        """
        The proxy client registers itself as a producer of the proxy server and
        vice versa.
        """
        addr = address.IPv4Address('TCP', '127.0.0.1', 0)
        server = portforward.ProxyFactory('127.0.0.1', 0).buildProtocol(addr)
        reactor = proto_helpers.MemoryReactor()
        server.reactor = reactor
        serverTransport = proto_helpers.StringTransport()
        server.makeConnection(serverTransport)
        self.assertEqual(len(reactor.tcpClients), 1)
        host, port, clientFactory, timeout, _ = reactor.tcpClients[0]
        self.assertIsInstance(clientFactory, portforward.ProxyClientFactory)
        client = clientFactory.buildProtocol(addr)
        clientTransport = proto_helpers.StringTransport()
        client.makeConnection(clientTransport)
        self.assertIs(clientTransport.producer, serverTransport)
        self.assertIs(serverTransport.producer, clientTransport)
        self.assertTrue(clientTransport.streaming)
        self.assertTrue(serverTransport.streaming)