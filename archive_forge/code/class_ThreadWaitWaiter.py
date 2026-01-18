import threading
import uuid
import fixtures
import testscenarios
from oslo_messaging._drivers import pool
from oslo_messaging.tests import utils as test_utils
class ThreadWaitWaiter(object):
    """A gross hack.

        Stub out the condition variable's wait() method and spin until it
        has been called by each thread.
        """

    def __init__(self, cond, n_threads, test):
        self.cond = cond
        self.test = test
        self.n_threads = n_threads
        self.n_waits = 0
        self.orig_wait = cond.wait

        def count_waits(**kwargs):
            self.n_waits += 1
            self.orig_wait(**kwargs)
        self.test.useFixture(fixtures.MockPatchObject(self.cond, 'wait', count_waits))

    def wait(self):
        while self.n_waits < self.n_threads:
            pass
        self.test.useFixture(fixtures.MockPatchObject(self.cond, 'wait', self.orig_wait))