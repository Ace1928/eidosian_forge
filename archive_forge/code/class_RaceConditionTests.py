import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
class RaceConditionTests(unittest.SynchronousTestCase):

    def setUp(self):
        self.threadpool = threadpool.ThreadPool(0, 10)
        self.event = threading.Event()
        self.threadpool.start()

        def done():
            self.threadpool.stop()
            del self.threadpool
        self.addCleanup(done)

    def getTimeout(self):
        """
        A reasonable number of seconds to time out.
        """
        return 5

    def test_synchronization(self):
        """
        If multiple threads are waiting on an event (via blocking on something
        in a callable passed to L{threadpool.ThreadPool.callInThread}), and
        there is spare capacity in the threadpool, sending another callable
        which will cause those to un-block to
        L{threadpool.ThreadPool.callInThread} will reliably run that callable
        and un-block the blocked threads promptly.

        @note: This is not really a unit test, it is a stress-test.  You may
            need to run it with C{trial -u} to fail reliably if there is a
            problem.  It is very hard to regression-test for this particular
            bug - one where the thread pool may consider itself as having
            "enough capacity" when it really needs to spin up a new thread if
            it possibly can - in a deterministic way, since the bug can only be
            provoked by subtle race conditions.
        """
        timeout = self.getTimeout()
        self.threadpool.callInThread(self.event.set)
        self.event.wait(timeout)
        self.event.clear()
        for i in range(3):
            self.threadpool.callInThread(self.event.wait)
        self.threadpool.callInThread(self.event.set)
        self.event.wait(timeout)
        if not self.event.isSet():
            self.event.set()
            self.fail("'set' did not run in thread; timed out waiting on 'wait'.")