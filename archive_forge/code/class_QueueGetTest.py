import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
class QueueGetTest(AsyncTestCase):

    @gen_test
    def test_blocking_get(self):
        q = queues.Queue()
        q.put_nowait(0)
        self.assertEqual(0, (yield q.get()))

    def test_nonblocking_get(self):
        q = queues.Queue()
        q.put_nowait(0)
        self.assertEqual(0, q.get_nowait())

    def test_nonblocking_get_exception(self):
        q = queues.Queue()
        self.assertRaises(queues.QueueEmpty, q.get_nowait)

    @gen_test
    def test_get_with_putters(self):
        q = queues.Queue(1)
        q.put_nowait(0)
        put = q.put(1)
        self.assertEqual(0, (yield q.get()))
        self.assertIsNone((yield put))

    @gen_test
    def test_blocking_get_wait(self):
        q = queues.Queue()
        q.put(0)
        self.io_loop.call_later(0.01, q.put_nowait, 1)
        self.io_loop.call_later(0.02, q.put_nowait, 2)
        self.assertEqual(0, (yield q.get(timeout=timedelta(seconds=1))))
        self.assertEqual(1, (yield q.get(timeout=timedelta(seconds=1))))

    @gen_test
    def test_get_timeout(self):
        q = queues.Queue()
        get_timeout = q.get(timeout=timedelta(seconds=0.01))
        get = q.get()
        with self.assertRaises(TimeoutError):
            yield get_timeout
        q.put_nowait(0)
        self.assertEqual(0, (yield get))

    @gen_test
    def test_get_timeout_preempted(self):
        q = queues.Queue()
        get = q.get(timeout=timedelta(seconds=0.01))
        q.put(0)
        yield gen.sleep(0.02)
        self.assertEqual(0, (yield get))

    @gen_test
    def test_get_clears_timed_out_putters(self):
        q = queues.Queue(1)
        putters = [q.put(i, timedelta(seconds=0.01)) for i in range(10)]
        put = q.put(10)
        self.assertEqual(10, len(q._putters))
        yield gen.sleep(0.02)
        self.assertEqual(10, len(q._putters))
        self.assertFalse(put.done())
        q.put(11)
        self.assertEqual(0, (yield q.get()))
        self.assertEqual(1, len(q._putters))
        for putter in putters[1:]:
            self.assertRaises(TimeoutError, putter.result)

    @gen_test
    def test_get_clears_timed_out_getters(self):
        q = queues.Queue()
        getters = [asyncio.ensure_future(q.get(timedelta(seconds=0.01))) for _ in range(10)]
        get = asyncio.ensure_future(q.get())
        self.assertEqual(11, len(q._getters))
        yield gen.sleep(0.02)
        self.assertEqual(11, len(q._getters))
        self.assertFalse(get.done())
        q.get()
        self.assertEqual(2, len(q._getters))
        for getter in getters:
            self.assertRaises(TimeoutError, getter.result)

    @gen_test
    def test_async_for(self):
        q = queues.Queue()
        for i in range(5):
            q.put(i)

        async def f():
            results = []
            async for i in q:
                results.append(i)
                if i == 4:
                    return results
        results = (yield f())
        self.assertEqual(results, list(range(5)))