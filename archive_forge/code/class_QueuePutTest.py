import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
class QueuePutTest(AsyncTestCase):

    @gen_test
    def test_blocking_put(self):
        q = queues.Queue()
        q.put(0)
        self.assertEqual(0, q.get_nowait())

    def test_nonblocking_put_exception(self):
        q = queues.Queue(1)
        q.put(0)
        self.assertRaises(queues.QueueFull, q.put_nowait, 1)

    @gen_test
    def test_put_with_getters(self):
        q = queues.Queue()
        get0 = q.get()
        get1 = q.get()
        yield q.put(0)
        self.assertEqual(0, (yield get0))
        yield q.put(1)
        self.assertEqual(1, (yield get1))

    @gen_test
    def test_nonblocking_put_with_getters(self):
        q = queues.Queue()
        get0 = q.get()
        get1 = q.get()
        q.put_nowait(0)
        yield gen.moment
        self.assertEqual(0, (yield get0))
        q.put_nowait(1)
        yield gen.moment
        self.assertEqual(1, (yield get1))

    @gen_test
    def test_blocking_put_wait(self):
        q = queues.Queue(1)
        q.put_nowait(0)

        def get_and_discard():
            q.get()
        self.io_loop.call_later(0.01, get_and_discard)
        self.io_loop.call_later(0.02, get_and_discard)
        futures = [q.put(0), q.put(1)]
        self.assertFalse(any((f.done() for f in futures)))
        yield futures

    @gen_test
    def test_put_timeout(self):
        q = queues.Queue(1)
        q.put_nowait(0)
        put_timeout = q.put(1, timeout=timedelta(seconds=0.01))
        put = q.put(2)
        with self.assertRaises(TimeoutError):
            yield put_timeout
        self.assertEqual(0, q.get_nowait())
        self.assertEqual(2, (yield q.get()))
        yield put

    @gen_test
    def test_put_timeout_preempted(self):
        q = queues.Queue(1)
        q.put_nowait(0)
        put = q.put(1, timeout=timedelta(seconds=0.01))
        q.get()
        yield gen.sleep(0.02)
        yield put

    @gen_test
    def test_put_clears_timed_out_putters(self):
        q = queues.Queue(1)
        putters = [q.put(i, timedelta(seconds=0.01)) for i in range(10)]
        put = q.put(10)
        self.assertEqual(10, len(q._putters))
        yield gen.sleep(0.02)
        self.assertEqual(10, len(q._putters))
        self.assertFalse(put.done())
        q.put(11)
        self.assertEqual(2, len(q._putters))
        for putter in putters[1:]:
            self.assertRaises(TimeoutError, putter.result)

    @gen_test
    def test_put_clears_timed_out_getters(self):
        q = queues.Queue()
        getters = [asyncio.ensure_future(q.get(timedelta(seconds=0.01))) for _ in range(10)]
        get = asyncio.ensure_future(q.get())
        q.get()
        self.assertEqual(12, len(q._getters))
        yield gen.sleep(0.02)
        self.assertEqual(12, len(q._getters))
        self.assertFalse(get.done())
        q.put(0)
        self.assertEqual(1, len(q._getters))
        self.assertEqual(0, (yield get))
        for getter in getters:
            self.assertRaises(TimeoutError, getter.result)

    @gen_test
    def test_float_maxsize(self):
        q = queues.Queue(maxsize=1.3)
        self.assertTrue(q.empty())
        self.assertFalse(q.full())
        q.put_nowait(0)
        q.put_nowait(1)
        self.assertFalse(q.empty())
        self.assertTrue(q.full())
        self.assertRaises(queues.QueueFull, q.put_nowait, 2)
        self.assertEqual(0, q.get_nowait())
        self.assertFalse(q.empty())
        self.assertFalse(q.full())
        yield q.put(2)
        put = q.put(3)
        self.assertFalse(put.done())
        self.assertEqual(1, (yield q.get()))
        yield put
        self.assertTrue(q.full())