import os
from queuelib.rrqueue import RoundRobinQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
class RRQueueStartDomainsTestMixin:

    def setUp(self):
        super().setUp()
        self.q = RoundRobinQueue(self.qfactory, start_domains=['1', '2'])

    def qfactory(self, key):
        raise NotImplementedError

    def test_push_pop_peek_key(self):
        self.q.push(b'c', '1')
        self.q.push(b'd', '2')
        self.assertEqual(self.q.peek(), b'd')
        self.assertEqual(self.q.pop(), b'd')
        self.assertEqual(self.q.peek(), b'c')
        self.assertEqual(self.q.pop(), b'c')
        self.assertEqual(self.q.peek(), None)
        self.assertEqual(self.q.pop(), None)

    def test_push_pop_peek_key_reversed(self):
        self.q.push(b'd', '2')
        self.q.push(b'c', '1')
        self.assertEqual(self.q.peek(), b'd')
        self.assertEqual(self.q.pop(), b'd')
        self.assertEqual(self.q.peek(), b'c')
        self.assertEqual(self.q.pop(), b'c')
        self.assertEqual(self.q.peek(), None)
        self.assertEqual(self.q.pop(), None)