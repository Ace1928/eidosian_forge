from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
class ReactorLoopTests(unittest.TestCase):

    def testFailure(self):

        def foo(x):
            raise TestException(x)
        lc = task.LoopingCall(foo, 'bar')
        return self.assertFailure(lc.start(0.1), TestException)

    def testFailAndStop(self):

        def foo(x):
            lc.stop()
            raise TestException(x)
        lc = task.LoopingCall(foo, 'bar')
        return self.assertFailure(lc.start(0.1), TestException)

    def testEveryIteration(self):
        ran = []

        def foo():
            ran.append(None)
            if len(ran) > 5:
                lc.stop()
        lc = task.LoopingCall(foo)
        d = lc.start(0)

        def stopped(ign):
            self.assertEqual(len(ran), 6)
        return d.addCallback(stopped)

    def testStopAtOnceLater(self):
        d = defer.Deferred()

        def foo():
            d.errback(failure.DefaultException('This task also should never get called.'))
        self._lc = task.LoopingCall(foo)
        self._lc.start(1, now=False)
        reactor.callLater(0, self._callback_for_testStopAtOnceLater, d)
        return d

    def _callback_for_testStopAtOnceLater(self, d):
        self._lc.stop()
        reactor.callLater(0, d.callback, 'success')

    def testWaitDeferred(self):
        timings = [0.2, 0.8]
        clock = task.Clock()

        def foo():
            d = defer.Deferred()
            d.addCallback(lambda _: lc.stop())
            clock.callLater(1, d.callback, None)
            return d
        lc = TestableLoopingCall(clock, foo)
        lc.start(0.2)
        clock.pump(timings)
        self.assertFalse(clock.calls)

    def testFailurePropagation(self):
        timings = [0.3]
        clock = task.Clock()

        def foo():
            d = defer.Deferred()
            clock.callLater(0.3, d.errback, TestException())
            return d
        lc = TestableLoopingCall(clock, foo)
        d = lc.start(1)
        self.assertFailure(d, TestException)
        clock.pump(timings)
        self.assertFalse(clock.calls)
        return d

    def test_deferredWithCount(self):
        """
        In the case that the function passed to L{LoopingCall.withCount}
        returns a deferred, which does not fire before the next interval
        elapses, the function should not be run again. And if a function call
        is skipped in this fashion, the appropriate count should be
        provided.
        """
        testClock = task.Clock()
        d = defer.Deferred()
        deferredCounts = []

        def countTracker(possibleCount):
            deferredCounts.append(possibleCount)
            if len(deferredCounts) == 1:
                return d
            else:
                return None
        lc = task.LoopingCall.withCount(countTracker)
        lc.clock = testClock
        d = lc.start(0.2, now=False)
        self.assertEqual(deferredCounts, [])
        testClock.pump([0.2, 0.4])
        self.assertEqual(len(deferredCounts), 1)
        d.callback(None)
        testClock.pump([0.2])
        self.assertEqual(len(deferredCounts), 2)
        self.assertEqual(deferredCounts, [1, 3])