import os
import sys
import time
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, threads
from twisted.python import failure, log, threadable, threadpool
from twisted.trial.unittest import TestCase
import time
import %(reactor)s
from twisted.internet import reactor
@skipIf(not interfaces.IReactorThreads(reactor, None), 'No thread support, nothing to test here.')
class DeferredResultTests(TestCase):
    """
    Test twisted.internet.threads.
    """

    def setUp(self):
        reactor.suggestThreadPoolSize(8)

    def tearDown(self):
        reactor.suggestThreadPoolSize(0)

    def test_callMultiple(self):
        """
        L{threads.callMultipleInThread} calls multiple functions in a thread.
        """
        L = []
        N = 10
        d = defer.Deferred()

        def finished():
            self.assertEqual(L, list(range(N)))
            d.callback(None)
        threads.callMultipleInThread([(L.append, (i,), {}) for i in range(N)] + [(reactor.callFromThread, (finished,), {})])
        return d

    def test_deferredResult(self):
        """
        L{threads.deferToThread} executes the function passed, and correctly
        handles the positional and keyword arguments given.
        """
        d = threads.deferToThread(lambda x, y=5: x + y, 3, y=4)
        d.addCallback(self.assertEqual, 7)
        return d

    def test_deferredFailure(self):
        """
        Check that L{threads.deferToThread} return a failure object
        with an appropriate exception instance when the called
        function raises an exception.
        """

        class NewError(Exception):
            pass

        def raiseError():
            raise NewError()
        d = threads.deferToThread(raiseError)
        return self.assertFailure(d, NewError)

    def test_deferredFailureAfterSuccess(self):
        """
        Check that a successful L{threads.deferToThread} followed by a one
        that raises an exception correctly result as a failure.
        """
        d = threads.deferToThread(lambda: None)
        d.addCallback(lambda ign: threads.deferToThread(lambda: 1 // 0))
        return self.assertFailure(d, ZeroDivisionError)