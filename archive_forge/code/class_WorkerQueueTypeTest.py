import unittest
from pygame.threads import FuncResult, tmap, WorkerQueue, Empty, STOP
from pygame import threads, Surface, transform
import time
class WorkerQueueTypeTest(unittest.TestCase):

    def test_usage_with_different_functions(self):

        def f(x):
            return x + 1

        def f2(x):
            return x + 2
        wq = WorkerQueue()
        fr = FuncResult(f)
        fr2 = FuncResult(f2)
        wq.do(fr, 1)
        wq.do(fr2, 1)
        wq.wait()
        wq.stop()
        self.assertEqual(fr.result, 2)
        self.assertEqual(fr2.result, 3)

    def test_do(self):
        """Tests function placement on queue and execution after blocking function completion."""

    def test_stop(self):
        """Ensure stop() stops the worker queue"""
        wq = WorkerQueue()
        self.assertGreater(len(wq.pool), 0)
        for t in wq.pool:
            self.assertTrue(t.is_alive())
        for i in range(200):
            wq.do(lambda x: x + 1, i)
        wq.stop()
        for t in wq.pool:
            self.assertFalse(t.is_alive())
        self.assertIs(wq.queue.get(), STOP)

    def test_threadloop(self):
        wq = WorkerQueue(1)
        wq.do(wq.threadloop)
        l = []
        wq.do(l.append, 1)
        time.sleep(0.5)
        self.assertEqual(l[0], 1)
        wq.stop()
        self.assertFalse(wq.pool[0].is_alive())

    def test_wait(self):
        wq = WorkerQueue()
        for i in range(2000):
            wq.do(lambda x: x + 1, i)
        wq.wait()
        self.assertRaises(Empty, wq.queue.get_nowait)
        wq.stop()