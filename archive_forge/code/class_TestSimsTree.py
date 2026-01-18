import unittest
from collections import Counter
from low_index import *
class TestSimsTree(unittest.TestCase):

    def test_basic(self):
        t = SimsTree(2, 2, [], [])
        perm_reps = [node.permutation_rep() for node in t.list()]
        self.assertEqual(perm_reps, [[[0], [0]], [[0, 1], [1, 0]], [[1, 0], [0, 1]], [[1, 0], [1, 0]]])

    def test_figure_eight(self):
        t = SimsTree(2, 6, [[1, 1, 1, 2, -1, -2, -2, -1, 2], [2, 1, 1, 1, 2, -1, -2, -2, -1], [-1, 2, 1, 1, 1, 2, -1, -2, -2], [-2, -1, 2, 1, 1, 1, 2, -1, -2], [-2, -2, -1, 2, 1, 1, 1, 2, -1], [-1, -2, -2, -1, 2, 1, 1, 1, 2], [2, -1, -2, -2, -1, 2, 1, 1, 1], [1, 2, -1, -2, -2, -1, 2, 1, 1], [1, 1, 2, -1, -2, -2, -1, 2, 1], [1, 1, 1, 2, -1, -2, -2, -1, 2]], [])
        degrees = Counter([cover.degree for cover in t.list()])
        self.assertEqual(degrees[0], 0)
        self.assertEqual(degrees[1], 1)
        self.assertEqual(degrees[2], 1)
        self.assertEqual(degrees[3], 1)
        self.assertEqual(degrees[4], 2)
        self.assertEqual(degrees[5], 4)
        self.assertEqual(degrees[6], 11)
        self.assertEqual(degrees[7], 0)