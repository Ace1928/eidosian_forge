import unittest
from collections import Counter
from low_index import *
def _test_K11n34_7(self, num_threads):
    reps = permutation_reps(3, ['aaBcbbcAc'], ['aacAbCBBaCAAbbcBc'], 7, num_threads=num_threads)
    degrees = Counter([len(rep[0]) for rep in reps])
    self.assertEqual(degrees[1], 1)
    self.assertEqual(degrees[2], 1)
    self.assertEqual(degrees[3], 1)
    self.assertEqual(degrees[4], 1)
    self.assertEqual(degrees[5], 2)
    self.assertEqual(degrees[6], 16)
    self.assertEqual(degrees[7], 30)
    self.assertIn([[0], [0], [0]], reps)
    self.assertIn([[0, 3, 5, 4, 1, 2], [1, 0, 5, 2, 4, 3], [1, 4, 0, 3, 5, 2]], reps)