import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
class MemoryBackedTests(unittest.SynchronousTestCase):
    """
    Tests using L{PoolHelper} to deterministically test properties of the
    threadpool implementation.
    """

    def test_workBeforeStarting(self):
        """
        If a threadpool is told to do work before starting, then upon starting
        up, it will start enough workers to handle all of the enqueued work
        that it's been given.
        """
        helper = PoolHelper(self, 0, 10)
        n = 5
        for x in range(n):
            helper.threadpool.callInThread(lambda: None)
        helper.performAllCoordination()
        self.assertEqual(helper.workers, [])
        helper.threadpool.start()
        helper.performAllCoordination()
        self.assertEqual(len(helper.workers), n)

    def test_tooMuchWorkBeforeStarting(self):
        """
        If the amount of work before starting exceeds the maximum number of
        threads allowed to the threadpool, only the maximum count will be
        started.
        """
        helper = PoolHelper(self, 0, 10)
        n = 50
        for x in range(n):
            helper.threadpool.callInThread(lambda: None)
        helper.performAllCoordination()
        self.assertEqual(helper.workers, [])
        helper.threadpool.start()
        helper.performAllCoordination()
        self.assertEqual(len(helper.workers), helper.threadpool.max)