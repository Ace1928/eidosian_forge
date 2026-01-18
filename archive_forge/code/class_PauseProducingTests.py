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
class PauseProducingTests(TestCase):
    """
    Test some behaviors of pausing the production of a transport.
    """

    @skipIf(not interfaces.IReactorFDSet.providedBy(reactor), 'Reactor not providing IReactorFDSet')
    def test_pauseProducingInConnectionMade(self):
        """
        In C{connectionMade} of a client protocol, C{pauseProducing} used to be
        ignored: this test is here to ensure it's not ignored.
        """
        server = MyServerFactory()
        client = MyClientFactory()
        client.protocolConnectionMade = defer.Deferred()
        port = reactor.listenTCP(0, server, interface='127.0.0.1')
        self.addCleanup(port.stopListening)
        connector = reactor.connectTCP(port.getHost().host, port.getHost().port, client)
        self.addCleanup(connector.disconnect)

        def checkInConnectionMade(proto):
            tr = proto.transport
            self.assertIn(tr, reactor.getReaders() + reactor.getWriters())
            proto.transport.pauseProducing()
            self.assertNotIn(tr, reactor.getReaders() + reactor.getWriters())
            d = defer.Deferred()
            d.addCallback(checkAfterConnectionMade)
            reactor.callLater(0, d.callback, proto)
            return d

        def checkAfterConnectionMade(proto):
            tr = proto.transport
            self.assertNotIn(tr, reactor.getReaders() + reactor.getWriters())
        client.protocolConnectionMade.addCallback(checkInConnectionMade)
        return client.protocolConnectionMade