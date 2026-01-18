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
class DeferToThreadPoolTests(TestCase):
    """
    Test L{twisted.internet.threads.deferToThreadPool}.
    """

    def setUp(self):
        self.tp = threadpool.ThreadPool(0, 8)
        self.tp.start()

    def tearDown(self):
        self.tp.stop()

    def test_deferredResult(self):
        """
        L{threads.deferToThreadPool} executes the function passed, and
        correctly handles the positional and keyword arguments given.
        """
        d = threads.deferToThreadPool(reactor, self.tp, lambda x, y=5: x + y, 3, y=4)
        d.addCallback(self.assertEqual, 7)
        return d

    def test_deferredFailure(self):
        """
        Check that L{threads.deferToThreadPool} return a failure object with an
        appropriate exception instance when the called function raises an
        exception.
        """

        class NewError(Exception):
            pass

        def raiseError():
            raise NewError()
        d = threads.deferToThreadPool(reactor, self.tp, raiseError)
        return self.assertFailure(d, NewError)