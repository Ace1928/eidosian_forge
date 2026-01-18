import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
class ConditionTest(AsyncTestCase):

    def setUp(self):
        super().setUp()
        self.history = []

    def record_done(self, future, key):
        """Record the resolution of a Future returned by Condition.wait."""

        def callback(_):
            if not future.result():
                self.history.append('timeout')
            else:
                self.history.append(key)
        future.add_done_callback(callback)

    def loop_briefly(self):
        """Run all queued callbacks on the IOLoop.

        In these tests, this method is used after calling notify() to
        preserve the pre-5.0 behavior in which callbacks ran
        synchronously.
        """
        self.io_loop.add_callback(self.stop)
        self.wait()

    def test_repr(self):
        c = locks.Condition()
        self.assertIn('Condition', repr(c))
        self.assertNotIn('waiters', repr(c))
        c.wait()
        self.assertIn('waiters', repr(c))

    @gen_test
    def test_notify(self):
        c = locks.Condition()
        self.io_loop.call_later(0.01, c.notify)
        yield c.wait()

    def test_notify_1(self):
        c = locks.Condition()
        self.record_done(c.wait(), 'wait1')
        self.record_done(c.wait(), 'wait2')
        c.notify(1)
        self.loop_briefly()
        self.history.append('notify1')
        c.notify(1)
        self.loop_briefly()
        self.history.append('notify2')
        self.assertEqual(['wait1', 'notify1', 'wait2', 'notify2'], self.history)

    def test_notify_n(self):
        c = locks.Condition()
        for i in range(6):
            self.record_done(c.wait(), i)
        c.notify(3)
        self.loop_briefly()
        self.assertEqual(list(range(3)), self.history)
        c.notify(1)
        self.loop_briefly()
        self.assertEqual(list(range(4)), self.history)
        c.notify(2)
        self.loop_briefly()
        self.assertEqual(list(range(6)), self.history)

    def test_notify_all(self):
        c = locks.Condition()
        for i in range(4):
            self.record_done(c.wait(), i)
        c.notify_all()
        self.loop_briefly()
        self.history.append('notify_all')
        self.assertEqual(list(range(4)) + ['notify_all'], self.history)

    @gen_test
    def test_wait_timeout(self):
        c = locks.Condition()
        wait = c.wait(timedelta(seconds=0.01))
        self.io_loop.call_later(0.02, c.notify)
        yield gen.sleep(0.03)
        self.assertFalse((yield wait))

    @gen_test
    def test_wait_timeout_preempted(self):
        c = locks.Condition()
        self.io_loop.call_later(0.01, c.notify)
        wait = c.wait(timedelta(seconds=0.02))
        yield gen.sleep(0.03)
        yield wait

    @gen_test
    def test_notify_n_with_timeout(self):
        c = locks.Condition()
        self.record_done(c.wait(), 0)
        self.record_done(c.wait(timedelta(seconds=0.01)), 1)
        self.record_done(c.wait(), 2)
        self.record_done(c.wait(), 3)
        yield gen.sleep(0.02)
        self.assertEqual(['timeout'], self.history)
        c.notify(2)
        yield gen.sleep(0.01)
        self.assertEqual(['timeout', 0, 2], self.history)
        self.assertEqual(['timeout', 0, 2], self.history)
        c.notify()
        yield
        self.assertEqual(['timeout', 0, 2, 3], self.history)

    @gen_test
    def test_notify_all_with_timeout(self):
        c = locks.Condition()
        self.record_done(c.wait(), 0)
        self.record_done(c.wait(timedelta(seconds=0.01)), 1)
        self.record_done(c.wait(), 2)
        yield gen.sleep(0.02)
        self.assertEqual(['timeout'], self.history)
        c.notify_all()
        yield
        self.assertEqual(['timeout', 0, 2], self.history)

    @gen_test
    def test_nested_notify(self):
        c = locks.Condition()
        futures = [asyncio.ensure_future(c.wait()) for _ in range(3)]
        futures[1].add_done_callback(lambda _: c.notify())
        c.notify(2)
        yield
        self.assertTrue(all((f.done() for f in futures)))

    @gen_test
    def test_garbage_collection(self):
        c = locks.Condition()
        for _ in range(101):
            c.wait(timedelta(seconds=0.01))
        future = asyncio.ensure_future(c.wait())
        self.assertEqual(102, len(c._waiters))
        yield gen.sleep(0.02)
        self.assertEqual(1, len(c._waiters))
        self.assertFalse(future.done())
        c.notify()
        self.assertTrue(future.done())