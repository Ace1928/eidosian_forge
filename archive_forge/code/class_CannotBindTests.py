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
class CannotBindTests(TestCase):
    """
    Tests for correct behavior when a reactor cannot bind to the required TCP
    port.
    """

    def test_cannotBind(self):
        """
        L{IReactorTCP.listenTCP} raises L{error.CannotListenError} if the
        address to listen on is already in use.
        """
        f = MyServerFactory()
        p1 = reactor.listenTCP(0, f, interface='127.0.0.1')
        self.addCleanup(p1.stopListening)
        n = p1.getHost().port
        dest = p1.getHost()
        self.assertEqual(dest.type, 'TCP')
        self.assertEqual(dest.host, '127.0.0.1')
        self.assertEqual(dest.port, n)
        self.assertRaises(error.CannotListenError, reactor.listenTCP, n, f, interface='127.0.0.1')

    def _fireWhenDoneFunc(self, d, f):
        """Returns closure that when called calls f and then callbacks d."""

        @wraps(f)
        def newf(*args, **kw):
            rtn = f(*args, **kw)
            d.callback('')
            return rtn
        return newf

    def test_clientBind(self):
        """
        L{IReactorTCP.connectTCP} calls C{Factory.clientConnectionFailed} with
        L{error.ConnectBindError} if the bind address specified is already in
        use.
        """
        theDeferred = defer.Deferred()
        sf = MyServerFactory()
        sf.startFactory = self._fireWhenDoneFunc(theDeferred, sf.startFactory)
        p = reactor.listenTCP(0, sf, interface='127.0.0.1')
        self.addCleanup(p.stopListening)

        def _connect1(results):
            d = defer.Deferred()
            cf1 = MyClientFactory()
            cf1.buildProtocol = self._fireWhenDoneFunc(d, cf1.buildProtocol)
            reactor.connectTCP('127.0.0.1', p.getHost().port, cf1, bindAddress=('127.0.0.1', 0))
            d.addCallback(_conmade, cf1)
            return d

        def _conmade(results, cf1):
            d = defer.Deferred()
            cf1.protocol.connectionMade = self._fireWhenDoneFunc(d, cf1.protocol.connectionMade)
            d.addCallback(_check1connect2, cf1)
            return d

        def _check1connect2(results, cf1):
            self.assertEqual(cf1.protocol.made, 1)
            d1 = defer.Deferred()
            d2 = defer.Deferred()
            port = cf1.protocol.transport.getHost().port
            cf2 = MyClientFactory()
            cf2.clientConnectionFailed = self._fireWhenDoneFunc(d1, cf2.clientConnectionFailed)
            cf2.stopFactory = self._fireWhenDoneFunc(d2, cf2.stopFactory)
            reactor.connectTCP('127.0.0.1', p.getHost().port, cf2, bindAddress=('127.0.0.1', port))
            d1.addCallback(_check2failed, cf1, cf2)
            d2.addCallback(_check2stopped, cf1, cf2)
            dl = defer.DeferredList([d1, d2])
            dl.addCallback(_stop, cf1, cf2)
            return dl

        def _check2failed(results, cf1, cf2):
            self.assertEqual(cf2.failed, 1)
            cf2.reason.trap(error.ConnectBindError)
            self.assertTrue(cf2.reason.check(error.ConnectBindError))
            return results

        def _check2stopped(results, cf1, cf2):
            self.assertEqual(cf2.stopped, 1)
            return results

        def _stop(results, cf1, cf2):
            d = defer.Deferred()
            d.addCallback(_check1cleanup, cf1)
            cf1.stopFactory = self._fireWhenDoneFunc(d, cf1.stopFactory)
            cf1.protocol.transport.loseConnection()
            return d

        def _check1cleanup(results, cf1):
            self.assertEqual(cf1.stopped, 1)
        theDeferred.addCallback(_connect1)
        return theDeferred