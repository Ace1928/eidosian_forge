from hashlib import md5
from os import close, fstat, stat, unlink, urandom
from pprint import pformat
from socket import AF_INET, SOCK_STREAM, SOL_SOCKET, socket
from stat import S_IMODE
from struct import pack
from tempfile import mkstemp, mktemp
from typing import Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from twisted.internet import base, interfaces
from twisted.internet.address import UNIXAddress
from twisted.internet.defer import Deferred, fail, gatherResults
from twisted.internet.endpoints import UNIXClientEndpoint, UNIXServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, DatagramProtocol, ServerFactory
from twisted.internet.task import LoopingCall
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.log import addObserver, err, removeObserver
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
class UNIXAdoptStreamConnectionTestsBuilder(WriteSequenceTestsMixin, ReactorBuilder):
    requiredInterfaces = (IReactorFDSet, IReactorSocket, IReactorUNIX)

    def test_buildProtocolReturnsNone(self):
        """
        {IReactorSocket.adoptStreamConnection} returns None if the given
        factory's buildProtocol returns None.
        """
        reactor = self.buildReactor()
        from socket import socketpair

        class NoneFactory(ServerFactory):

            def buildProtocol(self, address):
                return None
        s1, s2 = socketpair(AF_UNIX, SOCK_STREAM)
        s1.setblocking(False)
        self.addCleanup(s1.close)
        self.addCleanup(s2.close)
        s1FD = s1.fileno()
        factory = NoneFactory()
        result = reactor.adoptStreamConnection(s1FD, AF_UNIX, factory)
        self.assertIsNone(result)

    def test_ServerAddressUNIX(self):
        """
        Helper method to test UNIX server addresses.
        """

        def connected(protocols):
            client, server, port = protocols
            try:
                portPath = _coerceToFilesystemEncoding('', port.getHost().name)
                self.assertEqual('<AccumulatingProtocol #%s on %s>' % (server.transport.sessionno, portPath), str(server.transport))
                self.assertEqual('AccumulatingProtocol,%s,%s' % (server.transport.sessionno, portPath), server.transport.logstr)
                peerAddress = server.factory.peerAddresses[0]
                self.assertIsInstance(peerAddress, UNIXAddress)
            finally:
                server.transport.loseConnection()
        reactor = self.buildReactor()
        d = self.getConnectedClientAndServer(reactor, interface=None, addressFamily=None)
        d.addCallback(connected)
        self.runReactor(reactor)

    def getConnectedClientAndServer(self, reactor, interface, addressFamily):
        """
        Return a L{Deferred} firing with a L{MyClientFactory} and
        L{MyServerFactory} connected pair, and the listening C{Port}. The
        particularity is that the server protocol has been obtained after doing
        a C{adoptStreamConnection} against the original server connection.
        """
        firstServer = MyServerFactory()
        firstServer.protocolConnectionMade = Deferred()
        server = MyServerFactory()
        server.protocolConnectionMade = Deferred()
        server.protocolConnectionLost = Deferred()
        client = MyClientFactory()
        client.protocolConnectionMade = Deferred()
        client.protocolConnectionLost = Deferred()
        path = mktemp(suffix='.sock', dir='.')
        port = reactor.listenUNIX(path, firstServer)

        def firstServerConnected(proto):
            reactor.removeReader(proto.transport)
            reactor.removeWriter(proto.transport)
            reactor.adoptStreamConnection(proto.transport.fileno(), AF_UNIX, server)
        firstServer.protocolConnectionMade.addCallback(firstServerConnected)
        lostDeferred = gatherResults([client.protocolConnectionLost, server.protocolConnectionLost])

        def stop(result):
            if reactor.running:
                reactor.stop()
            return result
        lostDeferred.addBoth(stop)
        deferred = Deferred()
        deferred.addErrback(stop)
        startDeferred = gatherResults([client.protocolConnectionMade, server.protocolConnectionMade])

        def start(protocols):
            client, server = protocols
            deferred.callback((client, server, port))
        startDeferred.addCallback(start)
        reactor.connectUNIX(port.getHost().name, client)
        return deferred