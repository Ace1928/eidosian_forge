from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def _stoppingTest(self, delay):
    ran = []

    def foo():
        ran.append(None)
    clock = task.Clock()
    lc = TestableLoopingCall(clock, foo)
    lc.start(delay, now=False)
    lc.stop()
    self.assertFalse(ran)
    self.assertFalse(clock.calls)