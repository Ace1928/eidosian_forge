import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
class IterationTimeoutTests(WarningCheckerTestCase):
    """
    Tests for the timeout argument L{PosixReactorBase.run} calls
    L{PosixReactorBase.doIteration} with in the presence of various delayed
    calls.
    """

    def _checkIterationTimeout(self, reactor):
        timeout = []
        reactor.iterationTimeout.addCallback(timeout.append)
        reactor.iterationTimeout.addCallback(lambda ignored: reactor.stop())
        reactor.run()
        return timeout[0]

    def test_noCalls(self):
        """
        If there are no delayed calls, C{doIteration} is called with a
        timeout of L{None}.
        """
        reactor = TimeoutReportReactor()
        timeout = self._checkIterationTimeout(reactor)
        self.assertIsNone(timeout)

    def test_delayedCall(self):
        """
        If there is a delayed call, C{doIteration} is called with a timeout
        which is the difference between the current time and the time at
        which that call is to run.
        """
        reactor = TimeoutReportReactor()
        reactor.callLater(100, lambda: None)
        timeout = self._checkIterationTimeout(reactor)
        self.assertEqual(timeout, 100)

    def test_timePasses(self):
        """
        If a delayed call is scheduled and then some time passes, the
        timeout passed to C{doIteration} is reduced by the amount of time
        which passed.
        """
        reactor = TimeoutReportReactor()
        reactor.callLater(100, lambda: None)
        reactor.now += 25
        timeout = self._checkIterationTimeout(reactor)
        self.assertEqual(timeout, 75)

    def test_multipleDelayedCalls(self):
        """
        If there are several delayed calls, C{doIteration} is called with a
        timeout which is the difference between the current time and the
        time at which the earlier of the two calls is to run.
        """
        reactor = TimeoutReportReactor()
        reactor.callLater(50, lambda: None)
        reactor.callLater(10, lambda: None)
        reactor.callLater(100, lambda: None)
        timeout = self._checkIterationTimeout(reactor)
        self.assertEqual(timeout, 10)

    def test_resetDelayedCall(self):
        """
        If a delayed call is reset, the timeout passed to C{doIteration} is
        based on the interval between the time when reset is called and the
        new delay of the call.
        """
        reactor = TimeoutReportReactor()
        call = reactor.callLater(50, lambda: None)
        reactor.now += 25
        call.reset(15)
        timeout = self._checkIterationTimeout(reactor)
        self.assertEqual(timeout, 15)

    def test_delayDelayedCall(self):
        """
        If a delayed call is re-delayed, the timeout passed to
        C{doIteration} is based on the remaining time before the call would
        have been made and the additional amount of time passed to the delay
        method.
        """
        reactor = TimeoutReportReactor()
        call = reactor.callLater(50, lambda: None)
        reactor.now += 10
        call.delay(20)
        timeout = self._checkIterationTimeout(reactor)
        self.assertEqual(timeout, 60)

    def test_cancelDelayedCall(self):
        """
        If the only delayed call is canceled, L{None} is the timeout passed
        to C{doIteration}.
        """
        reactor = TimeoutReportReactor()
        call = reactor.callLater(50, lambda: None)
        call.cancel()
        timeout = self._checkIterationTimeout(reactor)
        self.assertIsNone(timeout)