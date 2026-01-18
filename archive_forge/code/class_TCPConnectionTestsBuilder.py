import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
class TCPConnectionTestsBuilder(ReactorBuilder):
    """
    Builder defining tests relating to L{twisted.internet.tcp.Connection}.
    """
    requiredInterfaces = (IReactorTCP,)

    def test_stopStartReading(self):
        """
        This test verifies transport socket read state after multiple
        pause/resumeProducing calls.
        """
        sf = ServerFactory()
        reactor = sf.reactor = self.buildReactor()
        skippedReactors = ['Glib2Reactor', 'Gtk2Reactor']
        reactorClassName = reactor.__class__.__name__
        if reactorClassName in skippedReactors and platform.isWindows():
            raise SkipTest('This test is broken on gtk/glib under Windows.')
        sf.protocol = StopStartReadingProtocol
        sf.ready = Deferred()
        sf.stop = Deferred()
        p = reactor.listenTCP(0, sf)
        port = p.getHost().port

        def proceed(protos, port):
            """
            Send several IOCPReactor's buffers' worth of data.
            """
            self.assertTrue(protos[0])
            self.assertTrue(protos[1])
            protos = (protos[0][1], protos[1][1])
            protos[0].transport.write(b'x' * (2 * 4096) + b'y' * (2 * 4096))
            return sf.stop.addCallback(cleanup, protos, port).addCallback(lambda ign: reactor.stop())

        def cleanup(data, protos, port):
            """
            Make sure IOCPReactor didn't start several WSARecv operations
            that clobbered each other's results.
            """
            self.assertEqual(data, b'x' * (2 * 4096) + b'y' * (2 * 4096), 'did not get the right data')
            return DeferredList([maybeDeferred(protos[0].transport.loseConnection), maybeDeferred(protos[1].transport.loseConnection), maybeDeferred(port.stopListening)])
        cc = TCP4ClientEndpoint(reactor, '127.0.0.1', port)
        cf = ClientFactory()
        cf.protocol = Protocol
        d = DeferredList([cc.connect(cf), sf.ready]).addCallback(proceed, p)
        d.addErrback(log.err)
        self.runReactor(reactor)

    @oneTransportTest
    def test_resumeProducing(self, reactor, server):
        """
        When a L{Server} is connected, its C{resumeProducing} method adds it as
        a reader to the reactor.
        """
        server.pauseProducing()
        assertNotReading(self, reactor, server)
        server.resumeProducing()
        assertReading(self, reactor, server)

    @oneTransportTest
    def test_resumeProducingWhileDisconnecting(self, reactor, server):
        """
        When a L{Server} has already started disconnecting via
        C{loseConnection}, its C{resumeProducing} method does not add it as a
        reader to its reactor.
        """
        server.loseConnection()
        server.resumeProducing()
        assertNotReading(self, reactor, server)

    @oneTransportTest
    def test_resumeProducingWhileDisconnected(self, reactor, server):
        """
        When a L{Server} has already lost its connection, its
        C{resumeProducing} method does not add it as a reader to its reactor.
        """
        server.connectionLost(Failure(Exception('dummy')))
        assertNotReading(self, reactor, server)
        server.resumeProducing()
        assertNotReading(self, reactor, server)

    def test_connectionLostAfterPausedTransport(self):
        """
        Alice connects to Bob.  Alice writes some bytes and then shuts down the
        connection.  Bob receives the bytes from the connection and then pauses
        the transport object.  Shortly afterwards Bob resumes the transport
        object.  At that point, Bob is notified that the connection has been
        closed.

        This is no problem for most reactors.  The underlying event notification
        API will probably just remind them that the connection has been closed.
        It is a little tricky for win32eventreactor (MsgWaitForMultipleObjects).
        MsgWaitForMultipleObjects will only deliver the close notification once.
        The reactor needs to remember that notification until Bob resumes the
        transport.
        """

        class Pauser(ConnectableProtocol):

            def __init__(self):
                self.events = []

            def dataReceived(self, bytes):
                self.events.append('paused')
                self.transport.pauseProducing()
                self.reactor.callLater(0, self.resume)

            def resume(self):
                self.events.append('resumed')
                self.transport.resumeProducing()

            def connectionLost(self, reason):
                self.events.append('lost')
                ConnectableProtocol.connectionLost(self, reason)

        class Client(ConnectableProtocol):

            def connectionMade(self):
                self.transport.write(b'some bytes for you')
                self.transport.loseConnection()
        pauser = Pauser()
        runProtocolsWithReactor(self, pauser, Client(), TCPCreator())
        self.assertEqual(pauser.events, ['paused', 'resumed', 'lost'])

    def test_doubleHalfClose(self):
        """
        If one side half-closes its connection, and then the other side of the
        connection calls C{loseWriteConnection}, and then C{loseConnection} in
        {writeConnectionLost}, the connection is closed correctly.

        This rather obscure case used to fail (see ticket #3037).
        """

        @implementer(IHalfCloseableProtocol)
        class ListenerProtocol(ConnectableProtocol):

            def readConnectionLost(self):
                self.transport.loseWriteConnection()

            def writeConnectionLost(self):
                self.transport.loseConnection()

        class Client(ConnectableProtocol):

            def connectionMade(self):
                self.transport.loseConnection()
        runProtocolsWithReactor(self, ListenerProtocol(), Client(), TCPCreator())