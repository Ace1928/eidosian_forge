import queue
from threading import Thread
from timeit import default_timer as timer
from unittest import mock
import testtools
from keystoneauth1 import _fair_semaphore
class SemaphoreTests(testtools.TestCase):

    def _thread_worker(self):
        while True:
            self.q.get()
            with self.s:
                self.mock_payload.do_something()
            self.q.task_done()

    def _concurrency_core(self, concurrency, delay):
        self.s = _fair_semaphore.FairSemaphore(concurrency, delay)
        self.q = queue.Queue()
        for i in range(5):
            t = Thread(target=self._thread_worker)
            t.daemon = True
            t.start()
        for item in range(0, 10):
            self.q.put(item)
        self.q.join()

    def setUp(self):
        super(SemaphoreTests, self).setUp()
        self.mock_payload = mock.Mock()

    def test_semaphore_no_concurrency(self):
        start = timer()
        self._concurrency_core(None, 0.1)
        end = timer()
        self.assertTrue(end - start > 1.0)
        self.assertEqual(self.mock_payload.do_something.call_count, 10)

    def test_semaphore_single_concurrency(self):
        start = timer()
        self._concurrency_core(1, 0.1)
        end = timer()
        self.assertTrue(end - start > 1.0)
        self.assertEqual(self.mock_payload.do_something.call_count, 10)

    def test_semaphore_multiple_concurrency(self):
        start = timer()
        self._concurrency_core(5, 0.1)
        end = timer()
        self.assertTrue(end - start > 1.0)
        self.assertEqual(self.mock_payload.do_something.call_count, 10)

    def test_semaphore_fast_no_concurrency(self):
        self._concurrency_core(None, 0.0)

    def test_semaphore_fast_single_concurrency(self):
        self._concurrency_core(1, 0.0)

    def test_semaphore_fast_multiple_concurrency(self):
        self._concurrency_core(5, 0.0)