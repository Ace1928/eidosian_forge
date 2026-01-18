import socket
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import defer, error
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import DatagramProtocol
from twisted.internet.test.connectionmixins import LogObserverMixin, findFreePort
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python import context
from twisted.python.log import ILogContext, err
from twisted.test.test_udp import GoodClient, Server
from twisted.trial.unittest import SkipTest
class UDPPortTestsMixin:
    """
    Tests for L{IReactorUDP.listenUDP} and
    L{IReactorSocket.adoptDatagramPort}.
    """

    def test_interface(self):
        """
        L{IReactorUDP.listenUDP} returns an object providing L{IListeningPort}.
        """
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, DatagramProtocol())
        self.assertTrue(verifyObject(IListeningPort, port))

    def test_getHost(self):
        """
        L{IListeningPort.getHost} returns an L{IPv4Address} giving a
        dotted-quad of the IPv4 address the port is listening on as well as
        the port number.
        """
        host, portNumber = findFreePort(type=socket.SOCK_DGRAM)
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, DatagramProtocol(), port=portNumber, interface=host)
        self.assertEqual(port.getHost(), IPv4Address('UDP', host, portNumber))

    @skipWithoutIPv6
    def test_getHostIPv6(self):
        """
        L{IListeningPort.getHost} returns an L{IPv6Address} when listening on
        an IPv6 interface.
        """
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, DatagramProtocol(), interface='::1')
        addr = port.getHost()
        self.assertEqual(addr.host, '::1')
        self.assertIsInstance(addr, IPv6Address)

    def test_invalidInterface(self):
        """
        An L{InvalidAddressError} is raised when trying to listen on an address
        that isn't a valid IPv4 or IPv6 address.
        """
        reactor = self.buildReactor()
        self.assertRaises(error.InvalidAddressError, reactor.listenUDP, DatagramProtocol(), 0, interface='example.com')

    def test_logPrefix(self):
        """
        Datagram transports implement L{ILoggingContext.logPrefix} to return a
        message reflecting the protocol they are running.
        """

        class CustomLogPrefixDatagramProtocol(DatagramProtocol):

            def __init__(self, prefix):
                self._prefix = prefix
                self.system = Deferred()

            def logPrefix(self):
                return self._prefix

            def datagramReceived(self, bytes, addr):
                if self.system is not None:
                    system = self.system
                    self.system = None
                    system.callback(context.get(ILogContext)['system'])
        reactor = self.buildReactor()
        protocol = CustomLogPrefixDatagramProtocol('Custom Datagrams')
        d = protocol.system
        port = self.getListeningPort(reactor, protocol)
        address = port.getHost()

        def gotSystem(system):
            self.assertEqual('Custom Datagrams (UDP)', system)
        d.addCallback(gotSystem)
        d.addErrback(err)
        d.addCallback(lambda ignored: reactor.stop())
        port.write(b'some bytes', ('127.0.0.1', address.port))
        self.runReactor(reactor)

    def test_writeSequence(self):
        """
        Write a sequence of L{bytes} to a L{DatagramProtocol}.
        """

        class SimpleDatagramProtocol(DatagramProtocol):

            def __init__(self):
                self.defer = Deferred()

            def datagramReceived(self, data, addr):
                self.defer.callback(data)
        reactor = self.buildReactor()
        protocol = SimpleDatagramProtocol()
        defer = protocol.defer
        port = self.getListeningPort(reactor, protocol)
        address = port.getHost()
        dataToWrite = (b'some', b'bytes', b'to', b'write')

        def gotData(data):
            self.assertEqual(b''.join(dataToWrite), data)
        defer.addCallback(gotData)
        defer.addErrback(err)
        defer.addCallback(lambda ignored: reactor.stop())
        port.writeSequence(dataToWrite, ('127.0.0.1', address.port))
        self.runReactor(reactor)

    def test_str(self):
        """
        C{str()} on the listening port object includes the port number.
        """
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, DatagramProtocol())
        self.assertIn(str(port.getHost().port), str(port))

    def test_repr(self):
        """
        C{repr()} on the listening port object includes the port number.
        """
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, DatagramProtocol())
        self.assertIn(repr(port.getHost().port), str(port))

    @skipWithoutIPv6
    def test_writeToIPv6Interface(self):
        """
        Writing to an IPv6 UDP socket on the loopback interface succeeds.
        """
        reactor = self.buildReactor()
        server = Server()
        serverStarted = server.startedDeferred = defer.Deferred()
        self.getListeningPort(reactor, server, interface='::1')
        client = GoodClient()
        clientStarted = client.startedDeferred = defer.Deferred()
        self.getListeningPort(reactor, client, interface='::1')
        cAddr = client.transport.getHost()

        def cbClientStarted(ignored):
            """
            Send a datagram from the client once it's started.

            @param ignored: a list of C{[None, None]}, which is ignored
            @returns: a deferred which fires when the server has received a
                datagram.
            """
            client.transport.write(b'spam', ('::1', server.transport.getHost().port))
            serverReceived = server.packetReceived = defer.Deferred()
            return serverReceived

        def cbServerReceived(ignored):
            """
            Stop the reactor after a datagram is received.

            @param ignored: L{None}, which is ignored
            @returns: L{None}
            """
            reactor.stop()
        d = defer.gatherResults([serverStarted, clientStarted])
        d.addCallback(cbClientStarted)
        d.addCallback(cbServerReceived)
        d.addErrback(err)
        self.runReactor(reactor)
        packet = server.packets[0]
        self.assertEqual(packet, (b'spam', (cAddr.host, cAddr.port)))

    @skipWithoutIPv6
    def test_connectedWriteToIPv6Interface(self):
        """
        An IPv6 address can be passed as the C{interface} argument to
        L{listenUDP}. The resulting Port accepts IPv6 datagrams.
        """
        reactor = self.buildReactor()
        server = Server()
        serverStarted = server.startedDeferred = defer.Deferred()
        self.getListeningPort(reactor, server, interface='::1')
        client = GoodClient()
        clientStarted = client.startedDeferred = defer.Deferred()
        self.getListeningPort(reactor, client, interface='::1')
        cAddr = client.transport.getHost()

        def cbClientStarted(ignored):
            """
            Send a datagram from the client once it's started.

            @param ignored: a list of C{[None, None]}, which is ignored
            @returns: a deferred which fires when the server has received a
                datagram.
            """
            client.transport.connect('::1', server.transport.getHost().port)
            client.transport.write(b'spam')
            serverReceived = server.packetReceived = defer.Deferred()
            return serverReceived

        def cbServerReceived(ignored):
            """
            Stop the reactor after a datagram is received.

            @param ignored: L{None}, which is ignored
            @returns: L{None}
            """
            reactor.stop()
        d = defer.gatherResults([serverStarted, clientStarted])
        d.addCallback(cbClientStarted)
        d.addCallback(cbServerReceived)
        d.addErrback(err)
        self.runReactor(reactor)
        packet = server.packets[0]
        self.assertEqual(packet, (b'spam', (cAddr.host, cAddr.port)))

    def test_writingToHostnameRaisesInvalidAddressError(self):
        """
        Writing to a hostname instead of an IP address will raise an
        L{InvalidAddressError}.
        """
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, DatagramProtocol())
        self.assertRaises(error.InvalidAddressError, port.write, 'spam', ('example.invalid', 1))

    @skipWithoutIPv6
    def test_writingToIPv6OnIPv4RaisesInvalidAddressError(self):
        """
        Writing to an IPv6 address on an IPv4 socket will raise an
        L{InvalidAddressError}.
        """
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, DatagramProtocol(), interface='127.0.0.1')
        self.assertRaises(error.InvalidAddressError, port.write, 'spam', ('::1', 1))

    @skipWithoutIPv6
    def test_writingToIPv4OnIPv6RaisesInvalidAddressError(self):
        """
        Writing to an IPv6 address on an IPv4 socket will raise an
        L{InvalidAddressError}.
        """
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, DatagramProtocol(), interface='::1')
        self.assertRaises(error.InvalidAddressError, port.write, 'spam', ('127.0.0.1', 1))

    def test_connectingToHostnameRaisesInvalidAddressError(self):
        """
        Connecting to a hostname instead of an IP address will raise an
        L{InvalidAddressError}.
        """
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, DatagramProtocol())
        self.assertRaises(error.InvalidAddressError, port.connect, 'example.invalid', 1)

    def test_allowBroadcast(self):
        """
        L{IListeningPort.setBroadcastAllowed} sets broadcast to be allowed
        on the socket.
        """
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, DatagramProtocol())
        port.setBroadcastAllowed(True)
        self.assertTrue(port.getBroadcastAllowed())