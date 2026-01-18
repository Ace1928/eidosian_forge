import unittest
from collections import Counter
from low_index import *
def _test_o9_03127_9(self, use_long):
    relator = 'aabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbbaabbbaaabbbaabbbaabbaabbb'
    reps = permutation_reps(2, [] if use_long else [relator], [relator] if use_long else [], 4)
    degrees = Counter([len(rep[0]) for rep in reps])
    self.assertEqual(degrees[1], 1)
    self.assertEqual(degrees[2], 3)
    self.assertEqual(degrees[3], 2)
    self.assertEqual(degrees[4], 8)
    self.assertIn([[0, 1, 3, 2], [1, 3, 0, 2]], reps)