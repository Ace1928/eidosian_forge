import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
class DiskTestMixin:

    def test_nonserializable_object_one(self):
        self.assertRaises(TypeError, self.q.push, lambda x: x, 0)
        self.assertEqual(self.q.close(), [])

    def test_nonserializable_object_many_close(self):
        self.q.push(b'a', 3)
        self.q.push(b'b', 1)
        self.assertRaises(TypeError, self.q.push, lambda x: x, 0)
        self.q.push(b'c', 2)
        self.assertEqual(self.q.pop(), b'b')
        self.assertEqual(sorted(self.q.close()), [2, 3])

    def test_nonserializable_object_many_pop(self):
        self.q.push(b'a', 3)
        self.q.push(b'b', 1)
        self.assertRaises(TypeError, self.q.push, lambda x: x, 0)
        self.q.push(b'c', 2)
        self.assertEqual(self.q.pop(), b'b')
        self.assertEqual(self.q.pop(), b'c')
        self.assertEqual(self.q.pop(), b'a')
        self.assertEqual(self.q.pop(), None)
        self.assertEqual(self.q.close(), [])

    def test_reopen_with_prio(self):
        q1 = PriorityQueue(self.qfactory)
        q1.push(b'a', 3)
        q1.push(b'b', 1)
        q1.push(b'c', 2)
        active = q1.close()
        q2 = PriorityQueue(self.qfactory, startprios=active)
        self.assertEqual(q2.pop(), b'b')
        self.assertEqual(q2.pop(), b'c')
        self.assertEqual(q2.pop(), b'a')
        self.assertEqual(q2.close(), [])