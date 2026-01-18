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
class HalfCloseTests(TestCase):
    """Test half-closing connections."""

    def setUp(self):
        self.f = f = MyHCFactory()
        self.p = p = reactor.listenTCP(0, f, interface='127.0.0.1')
        self.addCleanup(p.stopListening)
        d = loopUntil(lambda: p.connected)
        self.cf = protocol.ClientCreator(reactor, MyHCProtocol)
        d.addCallback(lambda _: self.cf.connectTCP(p.getHost().host, p.getHost().port))
        d.addCallback(self._setUp)
        return d

    def _setUp(self, client):
        self.client = client
        self.clientProtoConnectionLost = self.client.closedDeferred = defer.Deferred()
        self.assertEqual(self.client.transport.connected, 1)
        return loopUntil(lambda: getattr(self.f, 'protocol', None) is not None)

    def tearDown(self):
        self.assertEqual(self.client.closed, 0)
        self.client.transport.loseConnection()
        d = defer.maybeDeferred(self.p.stopListening)
        d.addCallback(lambda ign: self.clientProtoConnectionLost)
        d.addCallback(self._tearDown)
        return d

    def _tearDown(self, ignored):
        self.assertEqual(self.client.closed, 1)
        self.assertEqual(self.f.protocol.closed, 0)
        d = defer.Deferred()

        def _connectionLost(reason):
            self.f.protocol.closed = 1
            d.callback(None)
        self.f.protocol.connectionLost = _connectionLost
        self.f.protocol.transport.loseConnection()
        d.addCallback(lambda x: self.assertEqual(self.f.protocol.closed, 1))
        return d

    def testCloseWriteCloser(self):
        client = self.client
        f = self.f
        t = client.transport
        t.write(b'hello')
        d = loopUntil(lambda: len(t._tempDataBuffer) == 0)

        def loseWrite(ignored):
            t.loseWriteConnection()
            return loopUntil(lambda: t._writeDisconnected)

        def check(ignored):
            self.assertFalse(client.closed)
            self.assertTrue(client.writeHalfClosed)
            self.assertFalse(client.readHalfClosed)
            return loopUntil(lambda: f.protocol.readHalfClosed)

        def write(ignored):
            w = client.transport.write
            w(b' world')
            w(b'lalala fooled you')
            self.assertEqual(0, len(client.transport._tempDataBuffer))
            self.assertEqual(f.protocol.data, b'hello')
            self.assertFalse(f.protocol.closed)
            self.assertTrue(f.protocol.readHalfClosed)
        return d.addCallback(loseWrite).addCallback(check).addCallback(write)

    def testWriteCloseNotification(self):
        f = self.f
        f.protocol.transport.loseWriteConnection()
        d = defer.gatherResults([loopUntil(lambda: f.protocol.writeHalfClosed), loopUntil(lambda: self.client.readHalfClosed)])
        d.addCallback(lambda _: self.assertEqual(f.protocol.readHalfClosed, False))
        return d