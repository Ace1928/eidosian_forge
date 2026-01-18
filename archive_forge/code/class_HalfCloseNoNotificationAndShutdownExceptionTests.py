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
class HalfCloseNoNotificationAndShutdownExceptionTests(TestCase):

    def setUp(self):
        self.f = f = MyServerFactory()
        self.f.protocolConnectionMade = defer.Deferred()
        self.p = p = reactor.listenTCP(0, f, interface='127.0.0.1')
        d = protocol.ClientCreator(reactor, AccumulatingProtocol).connectTCP(p.getHost().host, p.getHost().port)
        d.addCallback(self._gotClient)
        return d

    def _gotClient(self, client):
        self.client = client
        return self.f.protocolConnectionMade

    def tearDown(self):
        self.client.transport.loseConnection()
        return self.p.stopListening()

    def testNoNotification(self):
        """
        TCP protocols support half-close connections, but not all of them
        support being notified of write closes.  In this case, test that
        half-closing the connection causes the peer's connection to be
        closed.
        """
        self.client.transport.write(b'hello')
        self.client.transport.loseWriteConnection()
        self.f.protocol.closedDeferred = d = defer.Deferred()
        self.client.closedDeferred = d2 = defer.Deferred()
        d.addCallback(lambda x: self.assertEqual(self.f.protocol.data, b'hello'))
        d.addCallback(lambda x: self.assertTrue(self.f.protocol.closed))
        return defer.gatherResults([d, d2])

    def testShutdownException(self):
        """
        If the other side has already closed its connection,
        loseWriteConnection should pass silently.
        """
        self.f.protocol.transport.loseConnection()
        self.client.transport.write(b'X')
        self.client.transport.loseWriteConnection()
        self.f.protocol.closedDeferred = d = defer.Deferred()
        self.client.closedDeferred = d2 = defer.Deferred()
        d.addCallback(lambda x: self.assertTrue(self.f.protocol.closed))
        return defer.gatherResults([d, d2])