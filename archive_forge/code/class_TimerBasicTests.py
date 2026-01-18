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
class TimerBasicTests(TestCase):

    def testTimerRuns(self):
        d = defer.Deferred()
        self.t = internet.TimerService(1, d.callback, 'hello')
        self.t.startService()
        d.addCallback(self.assertEqual, 'hello')
        d.addCallback(lambda x: self.t.stopService())
        d.addCallback(lambda x: self.assertFalse(self.t.running))
        return d

    def tearDown(self):
        return self.t.stopService()

    def testTimerRestart(self):
        d1 = defer.Deferred()
        d2 = defer.Deferred()
        work = [(d2, 'bar'), (d1, 'foo')]

        def trigger():
            d, arg = work.pop()
            d.callback(arg)
        self.t = internet.TimerService(1, trigger)
        self.t.startService()

        def onFirstResult(result):
            self.assertEqual(result, 'foo')
            return self.t.stopService()

        def onFirstStop(ignored):
            self.assertFalse(self.t.running)
            self.t.startService()
            return d2

        def onSecondResult(result):
            self.assertEqual(result, 'bar')
            self.t.stopService()
        d1.addCallback(onFirstResult)
        d1.addCallback(onFirstStop)
        d1.addCallback(onSecondResult)
        return d1

    def testTimerLoops(self):
        l = []

        def trigger(data, number, d):
            l.append(data)
            if len(l) == number:
                d.callback(l)
        d = defer.Deferred()
        self.t = internet.TimerService(0.01, trigger, 'hello', 10, d)
        self.t.startService()
        d.addCallback(self.assertEqual, ['hello'] * 10)
        d.addCallback(lambda x: self.t.stopService())
        return d