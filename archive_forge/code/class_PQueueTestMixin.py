import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
class PQueueTestMixin:

    def setUp(self):
        QueuelibTestCase.setUp(self)
        self.q = PriorityQueue(self.qfactory)

    def qfactory(self, prio):
        raise NotImplementedError

    def test_len_nonzero(self):
        assert not self.q
        self.assertEqual(len(self.q), 0)
        self.q.push(b'a', 3)
        assert self.q
        self.q.push(b'b', 1)
        self.q.push(b'c', 2)
        self.q.push(b'd', 1)
        self.assertEqual(len(self.q), 4)
        self.q.pop()
        self.q.pop()
        self.q.pop()
        self.q.pop()
        assert not self.q
        self.assertEqual(len(self.q), 0)

    def test_close(self):
        self.q.push(b'a', 3)
        self.q.push(b'b', 1)
        self.q.push(b'c', 2)
        self.q.push(b'd', 1)
        iqueues = self.q.queues.values()
        self.assertEqual(sorted(self.q.close()), [1, 2, 3])
        assert all((q.closed for q in iqueues))

    def test_close_return_active(self):
        self.q.push(b'b', 1)
        self.q.push(b'c', 2)
        self.q.push(b'a', 3)
        self.q.pop()
        self.assertEqual(sorted(self.q.close()), [2, 3])

    def test_popped_internal_queues_closed(self):
        self.q.push(b'a', 3)
        self.q.push(b'b', 1)
        self.q.push(b'c', 2)
        p1queue = self.q.queues[1]
        self.assertEqual(self.q.pop(), b'b')
        self.q.close()
        assert p1queue.closed