import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
class QueueJoinTest(AsyncTestCase):
    queue_class = queues.Queue

    def test_task_done_underflow(self):
        q = self.queue_class()
        self.assertRaises(ValueError, q.task_done)

    @gen_test
    def test_task_done(self):
        q = self.queue_class()
        for i in range(100):
            q.put_nowait(i)
        self.accumulator = 0

        @gen.coroutine
        def worker():
            while True:
                item = (yield q.get())
                self.accumulator += item
                q.task_done()
                yield gen.sleep(random() * 0.01)
        worker()
        worker()
        yield q.join()
        self.assertEqual(sum(range(100)), self.accumulator)

    @gen_test
    def test_task_done_delay(self):
        q = self.queue_class()
        q.put_nowait(0)
        join = asyncio.ensure_future(q.join())
        self.assertFalse(join.done())
        yield q.get()
        self.assertFalse(join.done())
        yield gen.moment
        self.assertFalse(join.done())
        q.task_done()
        self.assertTrue(join.done())

    @gen_test
    def test_join_empty_queue(self):
        q = self.queue_class()
        yield q.join()
        yield q.join()

    @gen_test
    def test_join_timeout(self):
        q = self.queue_class()
        q.put(0)
        with self.assertRaises(TimeoutError):
            yield q.join(timeout=timedelta(seconds=0.01))