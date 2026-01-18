import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
class InternetTests(TestCase):

    def testTCP(self):
        s = service.MultiService()
        s.startService()
        factory = protocol.ServerFactory()
        factory.protocol = TestEcho
        TestEcho.d = defer.Deferred()
        t = internet.TCPServer(0, factory)
        t.setServiceParent(s)
        num = t._port.getHost().port
        factory = protocol.ClientFactory()
        factory.d = defer.Deferred()
        factory.protocol = Foo
        factory.line = None
        internet.TCPClient('127.0.0.1', num, factory).setServiceParent(s)
        factory.d.addCallback(self.assertEqual, b'lalala')
        factory.d.addCallback(lambda x: s.stopService())
        factory.d.addCallback(lambda x: TestEcho.d)
        return factory.d

    def test_UDP(self):
        """
        Test L{internet.UDPServer} with a random port: starting the service
        should give it valid port, and stopService should free it so that we
        can start a server on the same port again.
        """
        if not interfaces.IReactorUDP(reactor, None):
            raise SkipTest('This reactor does not support UDP sockets')
        p = protocol.DatagramProtocol()
        t = internet.UDPServer(0, p)
        t.startService()
        num = t._port.getHost().port
        self.assertNotEqual(num, 0)

        def onStop(ignored):
            t = internet.UDPServer(num, p)
            t.startService()
            return t.stopService()
        return defer.maybeDeferred(t.stopService).addCallback(onStop)

    def testPrivileged(self):
        factory = protocol.ServerFactory()
        factory.protocol = TestEcho
        TestEcho.d = defer.Deferred()
        t = internet.TCPServer(0, factory)
        t.privileged = 1
        t.privilegedStartService()
        num = t._port.getHost().port
        factory = protocol.ClientFactory()
        factory.d = defer.Deferred()
        factory.protocol = Foo
        factory.line = None
        c = internet.TCPClient('127.0.0.1', num, factory)
        c.startService()
        factory.d.addCallback(self.assertEqual, b'lalala')
        factory.d.addCallback(lambda x: c.stopService())
        factory.d.addCallback(lambda x: t.stopService())
        factory.d.addCallback(lambda x: TestEcho.d)
        return factory.d

    def testConnectionGettingRefused(self):
        factory = protocol.ServerFactory()
        factory.protocol = wire.Echo
        t = internet.TCPServer(0, factory)
        t.startService()
        num = t._port.getHost().port
        t.stopService()
        d = defer.Deferred()
        factory = protocol.ClientFactory()
        factory.clientConnectionFailed = lambda *args: d.callback(None)
        c = internet.TCPClient('127.0.0.1', num, factory)
        c.startService()
        return d

    @skipIf(not interfaces.IReactorUNIX(reactor, None), 'This reactor does not support UNIX domain sockets')
    def testUNIX(self):
        s = service.MultiService()
        s.startService()
        factory = protocol.ServerFactory()
        factory.protocol = TestEcho
        TestEcho.d = defer.Deferred()
        t = internet.UNIXServer('echo.skt', factory)
        t.setServiceParent(s)
        factory = protocol.ClientFactory()
        factory.protocol = Foo
        factory.d = defer.Deferred()
        factory.line = None
        internet.UNIXClient('echo.skt', factory).setServiceParent(s)
        factory.d.addCallback(self.assertEqual, b'lalala')
        factory.d.addCallback(lambda x: s.stopService())
        factory.d.addCallback(lambda x: TestEcho.d)
        factory.d.addCallback(self._cbTestUnix, factory, s)
        return factory.d

    def _cbTestUnix(self, ignored, factory, s):
        TestEcho.d = defer.Deferred()
        factory.line = None
        factory.d = defer.Deferred()
        s.startService()
        factory.d.addCallback(self.assertEqual, b'lalala')
        factory.d.addCallback(lambda x: s.stopService())
        factory.d.addCallback(lambda x: TestEcho.d)
        return factory.d

    @skipIf(not interfaces.IReactorUNIX(reactor, None), 'This reactor does not support UNIX domain sockets')
    def testVolatile(self):
        factory = protocol.ServerFactory()
        factory.protocol = wire.Echo
        t = internet.UNIXServer('echo.skt', factory)
        t.startService()
        self.failIfIdentical(t._port, None)
        t1 = copy.copy(t)
        self.assertIsNone(t1._port)
        t.stopService()
        self.assertIsNone(t._port)
        self.assertFalse(t.running)
        factory = protocol.ClientFactory()
        factory.protocol = wire.Echo
        t = internet.UNIXClient('echo.skt', factory)
        t.startService()
        self.failIfIdentical(t._connection, None)
        t1 = copy.copy(t)
        self.assertIsNone(t1._connection)
        t.stopService()
        self.assertIsNone(t._connection)
        self.assertFalse(t.running)

    @skipIf(not interfaces.IReactorUNIX(reactor, None), 'This reactor does not support UNIX domain sockets')
    def testStoppingServer(self):
        factory = protocol.ServerFactory()
        factory.protocol = wire.Echo
        t = internet.UNIXServer('echo.skt', factory)
        t.startService()
        t.stopService()
        self.assertFalse(t.running)
        factory = protocol.ClientFactory()
        d = defer.Deferred()
        factory.clientConnectionFailed = lambda *args: d.callback(None)
        reactor.connectUNIX('echo.skt', factory)
        return d

    def testPickledTimer(self):
        target = TimerTarget()
        t0 = internet.TimerService(1, target.append, 'hello')
        t0.startService()
        s = pickle.dumps(t0)
        t0.stopService()
        t = pickle.loads(s)
        self.assertFalse(t.running)

    def testBrokenTimer(self):
        d = defer.Deferred()
        t = internet.TimerService(1, lambda: 1 // 0)
        oldFailed = t._failed

        def _failed(why):
            oldFailed(why)
            d.callback(None)
        t._failed = _failed
        t.startService()
        d.addCallback(lambda x: t.stopService)
        d.addCallback(lambda x: self.assertEqual([ZeroDivisionError], [o.value.__class__ for o in self.flushLoggedErrors(ZeroDivisionError)]))
        return d

    def test_everythingThere(self):
        """
        L{twisted.application.internet} dynamically defines a set of
        L{service.Service} subclasses that in general have corresponding
        reactor.listenXXX or reactor.connectXXX calls.
        """
        trans = ['TCP', 'UNIX', 'SSL', 'UDP', 'UNIXDatagram', 'Multicast']
        for tran in trans[:]:
            if not getattr(interfaces, 'IReactor' + tran)(reactor, None):
                trans.remove(tran)
        for tran in trans:
            for side in ['Server', 'Client']:
                if tran == 'Multicast' and side == 'Client':
                    continue
                if tran == 'UDP' and side == 'Client':
                    continue
                self.assertTrue(hasattr(internet, tran + side))
                method = getattr(internet, tran + side).method
                prefix = {'Server': 'listen', 'Client': 'connect'}[side]
                self.assertTrue(hasattr(reactor, prefix + method) or (prefix == 'connect' and method == 'UDP'))
                o = getattr(internet, tran + side)()
                self.assertEqual(service.IService(o), o)

    def test_importAll(self):
        """
        L{twisted.application.internet} dynamically defines L{service.Service}
        subclasses. This test ensures that the subclasses exposed by C{__all__}
        are valid attributes of the module.
        """
        for cls in internet.__all__:
            self.assertTrue(hasattr(internet, cls), f'{cls} not importable from twisted.application.internet')

    def test_reactorParametrizationInServer(self):
        """
        L{internet._AbstractServer} supports a C{reactor} keyword argument
        that can be used to parametrize the reactor used to listen for
        connections.
        """
        reactor = MemoryReactor()
        factory = object()
        t = internet.TCPServer(1234, factory, reactor=reactor)
        t.startService()
        self.assertEqual(reactor.tcpServers.pop()[:2], (1234, factory))

    def test_reactorParametrizationInClient(self):
        """
        L{internet._AbstractClient} supports a C{reactor} keyword arguments
        that can be used to parametrize the reactor used to create new client
        connections.
        """
        reactor = MemoryReactor()
        factory = protocol.ClientFactory()
        t = internet.TCPClient('127.0.0.1', 1234, factory, reactor=reactor)
        t.startService()
        self.assertEqual(reactor.tcpClients.pop()[:3], ('127.0.0.1', 1234, factory))

    def test_reactorParametrizationInServerMultipleStart(self):
        """
        Like L{test_reactorParametrizationInServer}, but stop and restart the
        service and check that the given reactor is still used.
        """
        reactor = MemoryReactor()
        factory = protocol.Factory()
        t = internet.TCPServer(1234, factory, reactor=reactor)
        t.startService()
        self.assertEqual(reactor.tcpServers.pop()[:2], (1234, factory))
        t.stopService()
        t.startService()
        self.assertEqual(reactor.tcpServers.pop()[:2], (1234, factory))

    def test_reactorParametrizationInClientMultipleStart(self):
        """
        Like L{test_reactorParametrizationInClient}, but stop and restart the
        service and check that the given reactor is still used.
        """
        reactor = MemoryReactor()
        factory = protocol.ClientFactory()
        t = internet.TCPClient('127.0.0.1', 1234, factory, reactor=reactor)
        t.startService()
        self.assertEqual(reactor.tcpClients.pop()[:3], ('127.0.0.1', 1234, factory))
        t.stopService()
        t.startService()
        self.assertEqual(reactor.tcpClients.pop()[:3], ('127.0.0.1', 1234, factory))