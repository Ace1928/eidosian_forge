import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
class AddressTests(TestCase):
    """
    Tests for address-related interactions with client and server protocols.
    """

    def setUp(self):
        """
        Create a port and connected client/server pair which can be used
        to test factory behavior related to addresses.

        @return: A L{defer.Deferred} which will be called back when both the
            client and server protocols have received their connection made
            callback.
        """

        class RememberingWrapper(protocol.ClientFactory):
            """
            Simple wrapper factory which records the addresses which are
            passed to its L{buildProtocol} method and delegates actual
            protocol creation to another factory.

            @ivar addresses: A list of the objects passed to buildProtocol.
            @ivar factory: The wrapped factory to which protocol creation is
                delegated.
            """

            def __init__(self, factory):
                self.addresses = []
                self.factory = factory

            def buildProtocol(self, addr):
                """
                Append the given address to C{self.addresses} and forward
                the call to C{self.factory}.
                """
                self.addresses.append(addr)
                return self.factory.buildProtocol(addr)
        self.server = MyServerFactory()
        self.serverConnMade = self.server.protocolConnectionMade = defer.Deferred()
        self.serverConnLost = self.server.protocolConnectionLost = defer.Deferred()
        self.serverWrapper = RememberingWrapper(self.server)
        self.client = MyClientFactory()
        self.clientConnMade = self.client.protocolConnectionMade = defer.Deferred()
        self.clientConnLost = self.client.protocolConnectionLost = defer.Deferred()
        self.clientWrapper = RememberingWrapper(self.client)
        self.port = reactor.listenTCP(0, self.serverWrapper, interface='127.0.0.1')
        self.connector = reactor.connectTCP(self.port.getHost().host, self.port.getHost().port, self.clientWrapper)
        return defer.gatherResults([self.serverConnMade, self.clientConnMade])

    def tearDown(self):
        """
        Disconnect the client/server pair and shutdown the port created in
        L{setUp}.
        """
        self.connector.disconnect()
        return defer.gatherResults([self.serverConnLost, self.clientConnLost, defer.maybeDeferred(self.port.stopListening)])

    def test_buildProtocolClient(self):
        """
        L{ClientFactory.buildProtocol} should be invoked with the address of
        the server to which a connection has been established, which should
        be the same as the address reported by the C{getHost} method of the
        transport of the server protocol and as the C{getPeer} method of the
        transport of the client protocol.
        """
        serverHost = self.server.protocol.transport.getHost()
        clientPeer = self.client.protocol.transport.getPeer()
        self.assertEqual(self.clientWrapper.addresses, [IPv4Address('TCP', serverHost.host, serverHost.port)])
        self.assertEqual(self.clientWrapper.addresses, [IPv4Address('TCP', clientPeer.host, clientPeer.port)])