import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
class OrderedSetTest(test.TestCase):

    def test_pickleable(self):
        items = [10, 9, 8, 7]
        s = sets.OrderedSet(items)
        self.assertEqual(items, list(s))
        s_bin = pickle.dumps(s)
        s2 = pickle.loads(s_bin)
        self.assertEqual(s, s2)
        self.assertEqual(items, list(s2))

    def test_retain_ordering(self):
        items = [10, 9, 8, 7]
        s = sets.OrderedSet(iter(items))
        self.assertEqual(items, list(s))

    def test_retain_duplicate_ordering(self):
        items = [10, 9, 10, 8, 9, 7, 8]
        s = sets.OrderedSet(iter(items))
        self.assertEqual([10, 9, 8, 7], list(s))

    def test_length(self):
        items = [10, 9, 8, 7]
        s = sets.OrderedSet(iter(items))
        self.assertEqual(4, len(s))

    def test_duplicate_length(self):
        items = [10, 9, 10, 8, 9, 7, 8]
        s = sets.OrderedSet(iter(items))
        self.assertEqual(4, len(s))

    def test_contains(self):
        items = [10, 9, 8, 7]
        s = sets.OrderedSet(iter(items))
        for i in items:
            self.assertIn(i, s)

    def test_copy(self):
        items = [10, 9, 8, 7]
        s = sets.OrderedSet(iter(items))
        s2 = s.copy()
        self.assertEqual(s, s2)
        self.assertEqual(items, list(s2))

    def test_empty_intersection(self):
        s = sets.OrderedSet([1, 2, 3])
        es = set(s)
        self.assertEqual(es.intersection(), s.intersection())

    def test_intersection(self):
        s = sets.OrderedSet([1, 2, 3])
        s2 = sets.OrderedSet([2, 3, 4, 5])
        es = set(s)
        es2 = set(s2)
        self.assertEqual(es.intersection(es2), s.intersection(s2))
        self.assertEqual(es2.intersection(s), s2.intersection(s))

    def test_multi_intersection(self):
        s = sets.OrderedSet([1, 2, 3])
        s2 = sets.OrderedSet([2, 3, 4, 5])
        s3 = sets.OrderedSet([1, 2])
        es = set(s)
        es2 = set(s2)
        es3 = set(s3)
        self.assertEqual(es.intersection(s2, s3), s.intersection(s2, s3))
        self.assertEqual(es2.intersection(es3), s2.intersection(s3))

    def test_superset(self):
        s = sets.OrderedSet([1, 2, 3])
        s2 = sets.OrderedSet([2, 3])
        self.assertTrue(s.issuperset(s2))
        self.assertFalse(s.issubset(s2))

    def test_subset(self):
        s = sets.OrderedSet([1, 2, 3])
        s2 = sets.OrderedSet([2, 3])
        self.assertTrue(s2.issubset(s))
        self.assertFalse(s2.issuperset(s))

    def test_empty_difference(self):
        s = sets.OrderedSet([1, 2, 3])
        es = set(s)
        self.assertEqual(es.difference(), s.difference())

    def test_difference(self):
        s = sets.OrderedSet([1, 2, 3])
        s2 = sets.OrderedSet([2, 3])
        es = set(s)
        es2 = set(s2)
        self.assertEqual(es.difference(es2), s.difference(s2))
        self.assertEqual(es2.difference(es), s2.difference(s))

    def test_multi_difference(self):
        s = sets.OrderedSet([1, 2, 3])
        s2 = sets.OrderedSet([2, 3])
        s3 = sets.OrderedSet([3, 4, 5])
        es = set(s)
        es2 = set(s2)
        es3 = set(s3)
        self.assertEqual(es3.difference(es), s3.difference(s))
        self.assertEqual(es.difference(es3), s.difference(s3))
        self.assertEqual(es2.difference(es, es3), s2.difference(s, s3))

    def test_empty_union(self):
        s = sets.OrderedSet([1, 2, 3])
        es = set(s)
        self.assertEqual(es.union(), s.union())

    def test_union(self):
        s = sets.OrderedSet([1, 2, 3])
        s2 = sets.OrderedSet([2, 3, 4])
        es = set(s)
        es2 = set(s2)
        self.assertEqual(es.union(es2), s.union(s2))
        self.assertEqual(es2.union(es), s2.union(s))

    def test_multi_union(self):
        s = sets.OrderedSet([1, 2, 3])
        s2 = sets.OrderedSet([2, 3, 4])
        s3 = sets.OrderedSet([4, 5, 6])
        es = set(s)
        es2 = set(s2)
        es3 = set(s3)
        self.assertEqual(es.union(es2, es3), s.union(s2, s3))