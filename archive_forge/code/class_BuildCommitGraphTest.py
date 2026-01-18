from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore
from ..objects import Blob
from .utils import build_commit_graph, make_object
class BuildCommitGraphTest(TestCase):

    def setUp(self):
        super().setUp()
        self.store = MemoryObjectStore()

    def test_linear(self):
        c1, c2 = build_commit_graph(self.store, [[1], [2, 1]])
        for obj_id in [c1.id, c2.id, c1.tree, c2.tree]:
            self.assertIn(obj_id, self.store)
        self.assertEqual([], c1.parents)
        self.assertEqual([c1.id], c2.parents)
        self.assertEqual(c1.tree, c2.tree)
        self.assertEqual([], self.store[c1.tree].items())
        self.assertGreater(c2.commit_time, c1.commit_time)

    def test_merge(self):
        c1, c2, c3, c4 = build_commit_graph(self.store, [[1], [2, 1], [3, 1], [4, 2, 3]])
        self.assertEqual([c2.id, c3.id], c4.parents)
        self.assertGreater(c4.commit_time, c2.commit_time)
        self.assertGreater(c4.commit_time, c3.commit_time)

    def test_missing_parent(self):
        self.assertRaises(ValueError, build_commit_graph, self.store, [[1], [3, 2], [2, 1]])

    def test_trees(self):
        a1 = make_object(Blob, data=b'aaa1')
        a2 = make_object(Blob, data=b'aaa2')
        c1, c2 = build_commit_graph(self.store, [[1], [2, 1]], trees={1: [(b'a', a1)], 2: [(b'a', a2, 33188)]})
        self.assertEqual((33188, a1.id), self.store[c1.tree][b'a'])
        self.assertEqual((33188, a2.id), self.store[c2.tree][b'a'])

    def test_attrs(self):
        c1, c2 = build_commit_graph(self.store, [[1], [2, 1]], attrs={1: {'message': b'Hooray!'}})
        self.assertEqual(b'Hooray!', c1.message)
        self.assertEqual(b'Commit 2', c2.message)

    def test_commit_time(self):
        c1, c2, c3 = build_commit_graph(self.store, [[1], [2, 1], [3, 2]], attrs={1: {'commit_time': 124}, 2: {'commit_time': 123}})
        self.assertEqual(124, c1.commit_time)
        self.assertEqual(123, c2.commit_time)
        self.assertTrue(c2.commit_time < c1.commit_time < c3.commit_time)