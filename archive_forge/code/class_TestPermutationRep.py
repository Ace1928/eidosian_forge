import unittest
from collections import Counter
from low_index import *
class TestPermutationRep(unittest.TestCase):

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

    def test_K11n34_7_single_threaded(self):
        self._test_K11n34_7(num_threads=1)

    def test_K11n34_7_multi_threaded(self):
        self._test_K11n34_7(num_threads=0)

    def test_K11n34_7_fixed_multi_threads(self):
        self._test_K11n34_7(num_threads=48)

    def test_K15n12345_7(self):
        reps = permutation_reps(3, ['aBcACAcb'], ['aBaCacBAcAbaBabaCAcAbaBaCacBAcAbaBabCAcAbABaCabABAbABaCabCAcAb'], 7)
        degrees = Counter([len(rep[0]) for rep in reps])
        self.assertEqual(degrees[1], 1)
        self.assertEqual(degrees[2], 1)
        self.assertEqual(degrees[3], 1)
        self.assertEqual(degrees[4], 1)
        self.assertEqual(degrees[5], 3)
        self.assertEqual(degrees[6], 11)
        self.assertEqual(degrees[7], 22)
        self.assertIn([[0], [0], [0]], reps)
        self.assertIn([[0, 2, 1, 3, 4, 6, 5], [0, 3, 4, 5, 1, 2, 6], [1, 2, 0, 5, 6, 3, 4]], reps)

    def test_o9_15405_9(self):
        reps = permutation_reps(2, [], ['aaaaabbbaabbbaaaaabbbaabbbaaaaaBBBBBBBB'], 9)
        degrees = Counter([len(rep[0]) for rep in reps])
        self.assertEqual(degrees[1], 1)
        self.assertEqual(degrees[2], 1)
        self.assertEqual(degrees[3], 1)
        self.assertEqual(degrees[4], 1)
        self.assertEqual(degrees[5], 3)
        self.assertEqual(degrees[6], 3)
        self.assertEqual(degrees[7], 9)
        self.assertEqual(degrees[8], 5)
        self.assertEqual(degrees[9], 14)
        self.assertIn([[0], [0]], reps)
        self.assertIn([[0, 2, 4, 1, 5, 3], [1, 0, 5, 4, 2, 3]], reps)

    def _test_o9_03127_9(self, use_long):
        relator = 'aabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbb'
        reps = permutation_reps(2, [] if use_long else [relator], [relator] if use_long else [], 4)
        degrees = Counter([len(rep[0]) for rep in reps])
        self.assertEqual(degrees[1], 1)
        self.assertEqual(degrees[2], 3)
        self.assertEqual(degrees[3], 2)
        self.assertEqual(degrees[4], 8)
        self.assertIn([[0, 1, 3, 2], [1, 3, 0, 2]], reps)

    def test_o9_03127_9_short(self):
        self._test_o9_03127_9(False)

    def test_o9_03127_9_long(self):
        self._test_o9_03127_9(True)