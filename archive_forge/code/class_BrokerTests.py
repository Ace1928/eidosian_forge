import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
class BrokerTests(unittest.TestCase):
    thunkResult = None

    def tearDown(self):
        try:
            os.unlink('None-None-TESTING.pub')
        except OSError:
            pass

    def thunkErrorBad(self, error):
        self.fail(f'This should cause a return value, not {error}')

    def thunkResultGood(self, result):
        self.thunkResult = result

    def thunkErrorGood(self, tb):
        pass

    def thunkResultBad(self, result):
        self.fail(f'This should cause an error, not {result}')

    def test_reference(self):
        c, s, pump = connectedServerAndClient(test=self)

        class X(pb.Referenceable):

            def remote_catch(self, arg):
                self.caught = arg

        class Y(pb.Referenceable):

            def remote_throw(self, a, b):
                a.callRemote('catch', b)
        s.setNameForLocal('y', Y())
        y = c.remoteForName('y')
        x = X()
        z = X()
        y.callRemote('throw', x, z)
        pump.pump()
        pump.pump()
        pump.pump()
        self.assertIs(x.caught, z, 'X should have caught Z')
        self.assertEqual(y.remoteMethod('throw'), y.remoteMethod('throw'))

    def test_result(self):
        c, s, pump = connectedServerAndClient(test=self)
        for x, y in ((c, s), (s, c)):
            foo = SimpleRemote()
            x.setNameForLocal('foo', foo)
            bar = y.remoteForName('foo')
            self.expectedThunkResult = 8
            bar.callRemote('thunk', self.expectedThunkResult - 1).addCallbacks(self.thunkResultGood, self.thunkErrorBad)
            pump.pump()
            pump.pump()
            self.assertEqual(self.thunkResult, self.expectedThunkResult, "result wasn't received.")

    def refcountResult(self, result):
        self.nestedRemote = result

    def test_tooManyRefs(self):
        l = []
        e = []
        c, s, pump = connectedServerAndClient(test=self)
        foo = NestedRemote()
        s.setNameForLocal('foo', foo)
        x = c.remoteForName('foo')
        for igno in range(pb.MAX_BROKER_REFS + 10):
            if s.transport.closed or c.transport.closed:
                break
            x.callRemote('getSimple').addCallbacks(l.append, e.append)
            pump.pump()
        expected = pb.MAX_BROKER_REFS - 1
        self.assertTrue(s.transport.closed, 'transport was not closed')
        self.assertEqual(len(l), expected, f'expected {expected} got {len(l)}')

    def test_copy(self):
        c, s, pump = connectedServerAndClient(test=self)
        foo = NestedCopy()
        s.setNameForLocal('foo', foo)
        x = c.remoteForName('foo')
        x.callRemote('getCopy').addCallbacks(self.thunkResultGood, self.thunkErrorBad)
        pump.pump()
        pump.pump()
        self.assertEqual(self.thunkResult.x, 1)
        self.assertEqual(self.thunkResult.y['Hello'], 'World')
        self.assertEqual(self.thunkResult.z[0], 'test')

    def test_observe(self):
        c, s, pump = connectedServerAndClient(test=self)
        a = Observable()
        b = Observer()
        s.setNameForLocal('a', a)
        ra = c.remoteForName('a')
        ra.callRemote('observe', b)
        pump.pump()
        a.notify(1)
        pump.pump()
        pump.pump()
        a.notify(10)
        pump.pump()
        pump.pump()
        self.assertIsNotNone(b.obj, "didn't notify")
        self.assertEqual(b.obj, 1, 'notified too much')

    def test_defer(self):
        c, s, pump = connectedServerAndClient(test=self)
        d = DeferredRemote()
        s.setNameForLocal('d', d)
        e = c.remoteForName('d')
        pump.pump()
        pump.pump()
        results = []
        e.callRemote('doItLater').addCallback(results.append)
        pump.pump()
        pump.pump()
        self.assertFalse(d.run, 'Deferred method run too early.')
        d.d.callback(5)
        self.assertEqual(d.run, 5, 'Deferred method run too late.')
        pump.pump()
        pump.pump()
        self.assertEqual(results[0], 6, 'Incorrect result.')

    def test_refcount(self):
        c, s, pump = connectedServerAndClient(test=self)
        foo = NestedRemote()
        s.setNameForLocal('foo', foo)
        bar = c.remoteForName('foo')
        bar.callRemote('getSimple').addCallbacks(self.refcountResult, self.thunkErrorBad)
        pump.pump()
        pump.pump()
        rluid = self.nestedRemote.luid
        self.assertIn(rluid, s.localObjects)
        del self.nestedRemote
        if sys.hexversion >= 33554432:
            gc.collect()
        pump.pump()
        pump.pump()
        pump.pump()
        self.assertNotIn(rluid, s.localObjects)

    def test_cache(self):
        c, s, pump = connectedServerAndClient(test=self)
        obj = NestedCache()
        obj2 = NestedComplicatedCache()
        vcc = obj2.c
        s.setNameForLocal('obj', obj)
        s.setNameForLocal('xxx', obj2)
        o2 = c.remoteForName('obj')
        o3 = c.remoteForName('xxx')
        coll = []
        o2.callRemote('getCache').addCallback(coll.append).addErrback(coll.append)
        o2.callRemote('getCache').addCallback(coll.append).addErrback(coll.append)
        complex = []
        o3.callRemote('getCache').addCallback(complex.append)
        o3.callRemote('getCache').addCallback(complex.append)
        pump.flush()
        self.assertEqual(complex[0].x, 1)
        self.assertEqual(complex[0].y, 2)
        self.assertEqual(complex[0].foo, 3)
        vcc.setFoo4()
        pump.flush()
        self.assertEqual(complex[0].foo, 4)
        self.assertEqual(len(coll), 2)
        cp = coll[0][0]
        self.assertIdentical(cp.checkMethod().__self__, cp, 'potential refcounting issue')
        self.assertIdentical(cp.checkSelf(), cp, 'other potential refcounting issue')
        col2 = []
        o2.callRemote('putCache', cp).addCallback(col2.append)
        pump.flush()
        self.assertTrue(col2[0])
        self.assertEqual(o2.remoteMethod('getCache'), o2.remoteMethod('getCache'))
        luid = cp.luid
        baroqueLuid = complex[0].luid
        self.assertIn(luid, s.remotelyCachedObjects, "remote cache doesn't have it")
        del coll
        del cp
        pump.flush()
        del complex
        del col2
        pump.flush()
        if sys.hexversion >= 33554432:
            gc.collect()
        pump.flush()
        self.assertNotIn(luid, s.remotelyCachedObjects, 'Server still had it after GC')
        self.assertNotIn(luid, c.locallyCachedObjects, 'Client still had it after GC')
        self.assertNotIn(baroqueLuid, s.remotelyCachedObjects, 'Server still had complex after GC')
        self.assertNotIn(baroqueLuid, c.locallyCachedObjects, 'Client still had complex after GC')
        self.assertIsNone(vcc.observer, 'observer was not removed')

    def test_publishable(self):
        try:
            os.unlink('None-None-TESTING.pub')
        except OSError:
            pass
        c, s, pump = connectedServerAndClient(test=self)
        foo = GetPublisher()
        s.setNameForLocal('foo', foo)
        bar = c.remoteForName('foo')
        accum = []
        bar.callRemote('getPub').addCallbacks(accum.append, self.thunkErrorBad)
        pump.flush()
        obj = accum.pop()
        self.assertEqual(obj.activateCalled, 1)
        self.assertEqual(obj.isActivated, 1)
        self.assertEqual(obj.yayIGotPublished, 1)
        self.assertEqual(obj._wasCleanWhenLoaded, 0)
        c, s, pump = connectedServerAndClient(test=self)
        s.setNameForLocal('foo', foo)
        bar = c.remoteForName('foo')
        bar.callRemote('getPub').addCallbacks(accum.append, self.thunkErrorBad)
        pump.flush()
        obj = accum.pop()
        self.assertEqual(obj._wasCleanWhenLoaded, 1)

    def gotCopy(self, val):
        self.thunkResult = val.id

    def test_factoryCopy(self):
        c, s, pump = connectedServerAndClient(test=self)
        ID = 99
        obj = NestedCopy()
        s.setNameForLocal('foo', obj)
        x = c.remoteForName('foo')
        x.callRemote('getFactory', ID).addCallbacks(self.gotCopy, self.thunkResultBad)
        pump.pump()
        pump.pump()
        pump.pump()
        self.assertEqual(self.thunkResult, ID, f'ID not correct on factory object {self.thunkResult}')