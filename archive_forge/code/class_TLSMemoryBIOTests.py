from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
class TLSMemoryBIOTests(TestCase):
    """
    Tests for the implementation of L{ISSLTransport} which runs over another
    L{ITransport}.
    """

    def test_interfaces(self):
        """
        L{TLSMemoryBIOProtocol} instances provide L{ISSLTransport} and
        L{ISystemHandle}.
        """
        proto = TLSMemoryBIOProtocol(None, None)
        self.assertTrue(ISSLTransport.providedBy(proto))
        self.assertTrue(ISystemHandle.providedBy(proto))

    def test_wrappedProtocolInterfaces(self):
        """
        L{TLSMemoryBIOProtocol} instances provide the interfaces provided by
        the transport they wrap.
        """

        class ITransport(Interface):
            pass

        class MyTransport:

            def write(self, data):
                pass
        clientFactory = ClientFactory()
        contextFactory = ClientTLSContext()
        wrapperFactory = TLSMemoryBIOFactory(contextFactory, True, clientFactory)
        transport = MyTransport()
        directlyProvides(transport, ITransport)
        tlsProtocol = TLSMemoryBIOProtocol(wrapperFactory, Protocol())
        tlsProtocol.makeConnection(transport)
        self.assertTrue(ITransport.providedBy(tlsProtocol))

    def test_getHandle(self):
        """
        L{TLSMemoryBIOProtocol.getHandle} returns the L{OpenSSL.SSL.Connection}
        instance it uses to actually implement TLS.

        This may seem odd.  In fact, it is.  The L{OpenSSL.SSL.Connection} is
        not actually the "system handle" here, nor even an object the reactor
        knows about directly.  However, L{twisted.internet.ssl.Certificate}'s
        C{peerFromTransport} and C{hostFromTransport} methods depend on being
        able to get an L{OpenSSL.SSL.Connection} object in order to work
        properly.  Implementing L{ISystemHandle.getHandle} like this is the
        easiest way for those APIs to be made to work.  If they are changed,
        then it may make sense to get rid of this implementation of
        L{ISystemHandle} and return the underlying socket instead.
        """
        factory = ClientFactory()
        contextFactory = ClientTLSContext()
        wrapperFactory = TLSMemoryBIOFactory(contextFactory, True, factory)
        proto = TLSMemoryBIOProtocol(wrapperFactory, Protocol())
        transport = StringTransport()
        proto.makeConnection(transport)
        self.assertIsInstance(proto.getHandle(), Connection)

    def test_makeConnection(self):
        """
        When L{TLSMemoryBIOProtocol} is connected to a transport, it connects
        the protocol it wraps to a transport.
        """
        clientProtocol = Protocol()
        clientFactory = ClientFactory()
        clientFactory.protocol = lambda: clientProtocol
        contextFactory = ClientTLSContext()
        wrapperFactory = TLSMemoryBIOFactory(contextFactory, True, clientFactory)
        sslProtocol = wrapperFactory.buildProtocol(None)
        transport = StringTransport()
        sslProtocol.makeConnection(transport)
        self.assertIsNotNone(clientProtocol.transport)
        self.assertIsNot(clientProtocol.transport, transport)
        self.assertIs(clientProtocol.transport, sslProtocol)

    def handshakeProtocols(self):
        """
        Start handshake between TLS client and server.
        """
        clientFactory = ClientFactory()
        clientFactory.protocol = Protocol
        clientContextFactory, handshakeDeferred = HandshakeCallbackContextFactory.factoryAndDeferred()
        wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
        sslClientProtocol = wrapperFactory.buildProtocol(None)
        serverFactory = ServerFactory()
        serverFactory.protocol = Protocol
        serverContextFactory = ServerTLSContext()
        wrapperFactory = TLSMemoryBIOFactory(serverContextFactory, False, serverFactory)
        sslServerProtocol = wrapperFactory.buildProtocol(None)
        connectionDeferred = loopbackAsync(sslServerProtocol, sslClientProtocol)
        return (sslClientProtocol, sslServerProtocol, handshakeDeferred, connectionDeferred)

    def test_handshake(self):
        """
        The TLS handshake is performed when L{TLSMemoryBIOProtocol} is
        connected to a transport.
        """
        tlsClient, tlsServer, handshakeDeferred, _ = self.handshakeProtocols()
        return handshakeDeferred

    def test_handshakeFailure(self):
        """
        L{TLSMemoryBIOProtocol} reports errors in the handshake process to the
        application-level protocol object using its C{connectionLost} method
        and disconnects the underlying transport.
        """
        clientConnectionLost = Deferred()
        clientFactory = ClientFactory()
        clientFactory.protocol = lambda: ConnectionLostNotifyingProtocol(clientConnectionLost)
        clientContextFactory = HandshakeCallbackContextFactory()
        wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
        sslClientProtocol = wrapperFactory.buildProtocol(None)
        serverConnectionLost = Deferred()
        serverFactory = ServerFactory()
        serverFactory.protocol = lambda: ConnectionLostNotifyingProtocol(serverConnectionLost)
        certificateData = FilePath(certPath).getContent()
        certificate = PrivateCertificate.loadPEM(certificateData)
        serverContextFactory = certificate.options(certificate)
        wrapperFactory = TLSMemoryBIOFactory(serverContextFactory, False, serverFactory)
        sslServerProtocol = wrapperFactory.buildProtocol(None)
        connectionDeferred = loopbackAsync(sslServerProtocol, sslClientProtocol)

        def cbConnectionLost(protocol):
            protocol.lostConnectionReason.trap(Error)
        clientConnectionLost.addCallback(cbConnectionLost)
        serverConnectionLost.addCallback(cbConnectionLost)
        return gatherResults([clientConnectionLost, serverConnectionLost, connectionDeferred])

    def test_getPeerCertificate(self):
        """
        L{TLSMemoryBIOProtocol.getPeerCertificate} returns the
        L{OpenSSL.crypto.X509} instance representing the peer's
        certificate.
        """
        clientFactory = ClientFactory()
        clientFactory.protocol = Protocol
        clientContextFactory, handshakeDeferred = HandshakeCallbackContextFactory.factoryAndDeferred()
        wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
        sslClientProtocol = wrapperFactory.buildProtocol(None)
        serverFactory = ServerFactory()
        serverFactory.protocol = Protocol
        serverContextFactory = ServerTLSContext()
        wrapperFactory = TLSMemoryBIOFactory(serverContextFactory, False, serverFactory)
        sslServerProtocol = wrapperFactory.buildProtocol(None)
        loopbackAsync(sslServerProtocol, sslClientProtocol)

        def cbHandshook(ignored):
            cert = sslClientProtocol.getPeerCertificate()
            self.assertIsInstance(cert, crypto.X509)
            self.assertEqual(cert.digest('sha256'), b'D6:F2:2C:74:3B:E2:5E:F9:CA:DA:47:08:14:78:20:75:78:95:9E:52:BD:D2:7C:77:DD:D4:EE:DE:33:BF:34:40')
        handshakeDeferred.addCallback(cbHandshook)
        return handshakeDeferred

    def test_writeAfterHandshake(self):
        """
        Bytes written to L{TLSMemoryBIOProtocol} before the handshake is
        complete are received by the protocol on the other side of the
        connection once the handshake succeeds.
        """
        data = b'some bytes'
        clientProtocol = Protocol()
        clientFactory = ClientFactory()
        clientFactory.protocol = lambda: clientProtocol
        clientContextFactory, handshakeDeferred = HandshakeCallbackContextFactory.factoryAndDeferred()
        wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
        sslClientProtocol = wrapperFactory.buildProtocol(None)
        serverProtocol = AccumulatingProtocol(len(data))
        serverFactory = ServerFactory()
        serverFactory.protocol = lambda: serverProtocol
        serverContextFactory = ServerTLSContext()
        wrapperFactory = TLSMemoryBIOFactory(serverContextFactory, False, serverFactory)
        sslServerProtocol = wrapperFactory.buildProtocol(None)
        connectionDeferred = loopbackAsync(sslServerProtocol, sslClientProtocol)

        def cbHandshook(ignored):
            clientProtocol.transport.write(data)
            return connectionDeferred
        handshakeDeferred.addCallback(cbHandshook)

        def cbDisconnected(ignored):
            self.assertEqual(b''.join(serverProtocol.received), data)
        handshakeDeferred.addCallback(cbDisconnected)
        return handshakeDeferred

    def writeBeforeHandshakeTest(self, sendingProtocol, data):
        """
        Run test where client sends data before handshake, given the sending
        protocol and expected bytes.
        """
        clientFactory = ClientFactory()
        clientFactory.protocol = sendingProtocol
        clientContextFactory, handshakeDeferred = HandshakeCallbackContextFactory.factoryAndDeferred()
        wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
        sslClientProtocol = wrapperFactory.buildProtocol(None)
        serverProtocol = AccumulatingProtocol(len(data))
        serverFactory = ServerFactory()
        serverFactory.protocol = lambda: serverProtocol
        serverContextFactory = ServerTLSContext()
        wrapperFactory = TLSMemoryBIOFactory(serverContextFactory, False, serverFactory)
        sslServerProtocol = wrapperFactory.buildProtocol(None)
        connectionDeferred = loopbackAsync(sslServerProtocol, sslClientProtocol)

        def cbConnectionDone(ignored):
            self.assertEqual(b''.join(serverProtocol.received), data)
        connectionDeferred.addCallback(cbConnectionDone)
        return connectionDeferred

    def test_writeBeforeHandshake(self):
        """
        Bytes written to L{TLSMemoryBIOProtocol} before the handshake is
        complete are received by the protocol on the other side of the
        connection once the handshake succeeds.
        """
        data = b'some bytes'

        class SimpleSendingProtocol(Protocol):

            def connectionMade(self):
                self.transport.write(data)
        return self.writeBeforeHandshakeTest(SimpleSendingProtocol, data)

    def test_writeSequence(self):
        """
        Bytes written to L{TLSMemoryBIOProtocol} with C{writeSequence} are
        received by the protocol on the other side of the connection.
        """
        data = b'some bytes'

        class SimpleSendingProtocol(Protocol):

            def connectionMade(self):
                self.transport.writeSequence(list(iterbytes(data)))
        return self.writeBeforeHandshakeTest(SimpleSendingProtocol, data)

    def test_writeAfterLoseConnection(self):
        """
        Bytes written to L{TLSMemoryBIOProtocol} after C{loseConnection} is
        called are not transmitted (unless there is a registered producer,
        which will be tested elsewhere).
        """
        data = b'some bytes'

        class SimpleSendingProtocol(Protocol):

            def connectionMade(self):
                self.transport.write(data)
                self.transport.loseConnection()
                self.transport.write(b'hello')
                self.transport.writeSequence([b'world'])
        return self.writeBeforeHandshakeTest(SimpleSendingProtocol, data)

    def test_writeUnicodeRaisesTypeError(self):
        """
        Writing C{unicode} to L{TLSMemoryBIOProtocol} throws a C{TypeError}.
        """
        notBytes = 'hello'
        result = []

        class SimpleSendingProtocol(Protocol):

            def connectionMade(self):
                try:
                    self.transport.write(notBytes)
                    self.transport.write(b'bytes')
                    self.transport.loseConnection()
                except TypeError:
                    result.append(True)
                    self.transport.abortConnection()

        def flush_logged_errors():
            self.assertEqual(len(self.flushLoggedErrors(ConnectionLost, TypeError)), 2)
        d = self.writeBeforeHandshakeTest(SimpleSendingProtocol, b'bytes')
        d.addBoth(lambda ign: self.assertEqual(result, [True]))
        d.addBoth(lambda ign: deferLater(reactor, 0, flush_logged_errors))
        return d

    def test_multipleWrites(self):
        """
        If multiple separate TLS messages are received in a single chunk from
        the underlying transport, all of the application bytes from each
        message are delivered to the application-level protocol.
        """
        data = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i']

        class SimpleSendingProtocol(Protocol):

            def connectionMade(self):
                for b in data:
                    self.transport.write(b)
        clientFactory = ClientFactory()
        clientFactory.protocol = SimpleSendingProtocol
        clientContextFactory = HandshakeCallbackContextFactory()
        wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
        sslClientProtocol = wrapperFactory.buildProtocol(None)
        serverProtocol = AccumulatingProtocol(sum(map(len, data)))
        serverFactory = ServerFactory()
        serverFactory.protocol = lambda: serverProtocol
        serverContextFactory = ServerTLSContext()
        wrapperFactory = TLSMemoryBIOFactory(serverContextFactory, False, serverFactory)
        sslServerProtocol = wrapperFactory.buildProtocol(None)
        connectionDeferred = loopbackAsync(sslServerProtocol, sslClientProtocol, collapsingPumpPolicy)

        def cbConnectionDone(ignored):
            self.assertEqual(b''.join(serverProtocol.received), b''.join(data))
        connectionDeferred.addCallback(cbConnectionDone)
        return connectionDeferred

    def hugeWrite(self, method=TLS_METHOD):
        """
        If a very long string is passed to L{TLSMemoryBIOProtocol.write}, any
        trailing part of it which cannot be send immediately is buffered and
        sent later.
        """
        data = b'some bytes'
        factor = 2 ** 20

        class SimpleSendingProtocol(Protocol):

            def connectionMade(self):
                self.transport.write(data * factor)
        clientFactory = ClientFactory()
        clientFactory.protocol = SimpleSendingProtocol
        clientContextFactory = HandshakeCallbackContextFactory(method=method)
        wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
        sslClientProtocol = wrapperFactory.buildProtocol(None)
        serverProtocol = AccumulatingProtocol(len(data) * factor)
        serverFactory = ServerFactory()
        serverFactory.protocol = lambda: serverProtocol
        serverContextFactory = ServerTLSContext(method=method)
        wrapperFactory = TLSMemoryBIOFactory(serverContextFactory, False, serverFactory)
        sslServerProtocol = wrapperFactory.buildProtocol(None)
        connectionDeferred = loopbackAsync(sslServerProtocol, sslClientProtocol)

        def cbConnectionDone(ignored):
            self.assertEqual(b''.join(serverProtocol.received), data * factor)
        connectionDeferred.addCallback(cbConnectionDone)
        return connectionDeferred

    def test_hugeWrite(self):
        return self.hugeWrite()

    def test_hugeWrite_TLSv1_2(self):
        return self.hugeWrite(method=TLSv1_2_METHOD)

    def test_disorderlyShutdown(self):
        """
        If a L{TLSMemoryBIOProtocol} loses its connection unexpectedly, this is
        reported to the application.
        """
        clientConnectionLost = Deferred()
        clientFactory = ClientFactory()
        clientFactory.protocol = lambda: ConnectionLostNotifyingProtocol(clientConnectionLost)
        clientContextFactory = HandshakeCallbackContextFactory()
        wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
        sslClientProtocol = wrapperFactory.buildProtocol(None)
        serverProtocol = Protocol()
        loopbackAsync(serverProtocol, sslClientProtocol)
        serverProtocol.transport.loseConnection()

        def cbDisconnected(clientProtocol):
            clientProtocol.lostConnectionReason.trap(Error, ConnectionLost)
        clientConnectionLost.addCallback(cbDisconnected)
        return clientConnectionLost

    def test_loseConnectionAfterHandshake(self):
        """
        L{TLSMemoryBIOProtocol.loseConnection} sends a TLS close alert and
        shuts down the underlying connection cleanly on both sides, after
        transmitting all buffered data.
        """

        class NotifyingProtocol(ConnectionLostNotifyingProtocol):

            def __init__(self, onConnectionLost):
                ConnectionLostNotifyingProtocol.__init__(self, onConnectionLost)
                self.data = []

            def dataReceived(self, data):
                self.data.append(data)
        clientConnectionLost = Deferred()
        clientFactory = ClientFactory()
        clientProtocol = NotifyingProtocol(clientConnectionLost)
        clientFactory.protocol = lambda: clientProtocol
        clientContextFactory, handshakeDeferred = HandshakeCallbackContextFactory.factoryAndDeferred()
        wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
        sslClientProtocol = wrapperFactory.buildProtocol(None)
        serverConnectionLost = Deferred()
        serverProtocol = NotifyingProtocol(serverConnectionLost)
        serverFactory = ServerFactory()
        serverFactory.protocol = lambda: serverProtocol
        serverContextFactory = ServerTLSContext()
        wrapperFactory = TLSMemoryBIOFactory(serverContextFactory, False, serverFactory)
        sslServerProtocol = wrapperFactory.buildProtocol(None)
        loopbackAsync(sslServerProtocol, sslClientProtocol)
        chunkOfBytes = b'123456890' * 100000

        def cbHandshake(ignored):
            clientProtocol.transport.write(chunkOfBytes)
            serverProtocol.transport.write(b'x')
            serverProtocol.transport.loseConnection()
            return gatherResults([clientConnectionLost, serverConnectionLost])
        handshakeDeferred.addCallback(cbHandshake)

        def cbConnectionDone(result):
            clientProtocol, serverProtocol = result
            clientProtocol.lostConnectionReason.trap(ConnectionDone)
            serverProtocol.lostConnectionReason.trap(ConnectionDone)
            self.assertEqual(b''.join(serverProtocol.data), chunkOfBytes)
            self.assertTrue(serverProtocol.transport.q.disconnect)
            self.assertTrue(clientProtocol.transport.q.disconnect)
        handshakeDeferred.addCallback(cbConnectionDone)
        return handshakeDeferred

    def test_connectionLostOnlyAfterUnderlyingCloses(self):
        """
        The user protocol's connectionLost is only called when transport
        underlying TLS is disconnected.
        """

        class LostProtocol(Protocol):
            disconnected = None

            def connectionLost(self, reason):
                self.disconnected = reason
        wrapperFactory = TLSMemoryBIOFactory(ClientTLSContext(), True, ClientFactory())
        protocol = LostProtocol()
        tlsProtocol = TLSMemoryBIOProtocol(wrapperFactory, protocol)
        transport = StringTransport()
        tlsProtocol.makeConnection(transport)
        tlsProtocol._tlsShutdownFinished(None)
        self.assertTrue(transport.disconnecting)
        self.assertIsNone(protocol.disconnected)
        tlsProtocol.connectionLost(Failure(ConnectionLost('ono')))
        self.assertTrue(protocol.disconnected.check(ConnectionLost))
        self.assertEqual(protocol.disconnected.value.args, ('ono',))

    def test_loseConnectionTwice(self):
        """
        If TLSMemoryBIOProtocol.loseConnection is called multiple times, all
        but the first call have no effect.
        """
        tlsClient, tlsServer, handshakeDeferred, disconnectDeferred = self.handshakeProtocols()
        self.successResultOf(handshakeDeferred)
        calls = []

        def _shutdownTLS(shutdown=tlsClient._shutdownTLS):
            calls.append(1)
            return shutdown()
        tlsClient._shutdownTLS = _shutdownTLS
        tlsClient.write(b'x')
        tlsClient.loseConnection()
        self.assertTrue(tlsClient.disconnecting)
        self.assertEqual(calls, [1])
        tlsClient.loseConnection()
        self.assertEqual(calls, [1])
        return disconnectDeferred

    def test_loseConnectionAfterConnectionLost(self):
        """
        If TLSMemoryBIOProtocol.loseConnection is called after connectionLost,
        it does nothing.
        """
        tlsClient, tlsServer, handshakeDeferred, disconnectDeferred = self.handshakeProtocols()
        calls = []

        def _shutdownTLS(shutdown=tlsClient._shutdownTLS):
            calls.append(1)
            return shutdown()
        tlsServer._shutdownTLS = _shutdownTLS
        tlsServer.write(b'x')
        tlsClient.loseConnection()

        def disconnected(_):
            self.assertEqual(calls, [1])
            tlsServer.loseConnection()
            self.assertEqual(calls, [1])
        disconnectDeferred.addCallback(disconnected)
        return disconnectDeferred

    def test_unexpectedEOF(self):
        """
        Unexpected disconnects get converted to ConnectionLost errors.
        """
        tlsClient, tlsServer, handshakeDeferred, disconnectDeferred = self.handshakeProtocols()
        serverProtocol = tlsServer.wrappedProtocol
        data = []
        reason = []
        serverProtocol.dataReceived = data.append
        serverProtocol.connectionLost = reason.append

        def handshakeDone(ign):
            tlsClient.write(b'hello')
            tlsClient.transport.loseConnection()
        handshakeDeferred.addCallback(handshakeDone)

        def disconnected(ign):
            self.assertTrue(reason[0].check(ConnectionLost), reason[0])
        disconnectDeferred.addCallback(disconnected)
        return disconnectDeferred

    def test_errorWriting(self):
        """
        Errors while writing cause the protocols to be disconnected.
        """
        tlsClient, tlsServer, handshakeDeferred, disconnectDeferred = self.handshakeProtocols()
        reason = []
        tlsClient.wrappedProtocol.connectionLost = reason.append

        class Wrapper:

            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __getattr__(self, attr):
                return getattr(self._wrapped, attr)

            def send(self, *args):
                raise Error([('SSL routines', '', 'this message is probably useless')])
        tlsClient._tlsConnection = Wrapper(tlsClient._tlsConnection)

        def handshakeDone(ign):
            tlsClient.write(b'hello')
        handshakeDeferred.addCallback(handshakeDone)

        def disconnected(ign):
            self.assertTrue(reason[0].check(Error), reason[0])
        disconnectDeferred.addCallback(disconnected)
        return disconnectDeferred

    def test_noCircularReferences(self):
        """
        TLSMemoryBIOProtocol doesn't leave circular references that keep
        it in memory after connection is closed.
        """

        def nObjectsOfType(type):
            """
            Return the number of instances of a given type in memory.

            @param type: Type whose instances to find.

            @return: The number of instances found.
            """
            return sum((1 for x in gc.get_objects() if isinstance(x, type)))
        self.addCleanup(gc.enable)
        gc.disable()

        class CloserProtocol(Protocol):

            def dataReceived(self, data):
                self.transport.loseConnection()

        class GreeterProtocol(Protocol):

            def connectionMade(self):
                self.transport.write(b'hello')
        origTLSProtos = nObjectsOfType(TLSMemoryBIOProtocol)
        origServerProtos = nObjectsOfType(CloserProtocol)
        authCert, serverCert = certificatesForAuthorityAndServer()
        serverFactory = TLSMemoryBIOFactory(serverCert.options(), False, Factory.forProtocol(CloserProtocol))
        clientFactory = TLSMemoryBIOFactory(optionsForClientTLS('example.com', trustRoot=authCert), True, Factory.forProtocol(GreeterProtocol))
        loopbackAsync(TLSMemoryBIOProtocol(serverFactory, CloserProtocol()), TLSMemoryBIOProtocol(clientFactory, GreeterProtocol()))
        newTLSProtos = nObjectsOfType(TLSMemoryBIOProtocol)
        newServerProtos = nObjectsOfType(CloserProtocol)
        self.assertEqual(newTLSProtos, origTLSProtos)
        self.assertEqual(newServerProtos, origServerProtos)