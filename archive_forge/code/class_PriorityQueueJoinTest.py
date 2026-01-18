import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
class PriorityQueueJoinTest(QueueJoinTest):
    queue_class = queues.PriorityQueue

    @gen_test
    def test_order(self):
        q = self.queue_class(maxsize=2)
        q.put_nowait((1, 'a'))
        q.put_nowait((0, 'b'))
        self.assertTrue(q.full())
        q.put((3, 'c'))
        q.put((2, 'd'))
        self.assertEqual((0, 'b'), q.get_nowait())
        self.assertEqual((1, 'a'), (yield q.get()))
        self.assertEqual((2, 'd'), q.get_nowait())
        self.assertEqual((3, 'c'), (yield q.get()))
        self.assertTrue(q.empty())