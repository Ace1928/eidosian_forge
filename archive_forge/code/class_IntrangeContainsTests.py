import unittest
from idna.intranges import intranges_from_list, intranges_contain, _encode_range
class IntrangeContainsTests(unittest.TestCase):

    def _test_containment(self, ints, disjoint_ints):
        ranges = intranges_from_list(ints)
        for int_ in ints:
            assert intranges_contain(int_, ranges)
        for int_ in disjoint_ints:
            assert not intranges_contain(int_, ranges)

    def test_simple(self):
        self._test_containment(range(10, 20), [2, 3, 68, 3893])

    def test_skips(self):
        self._test_containment([0, 2, 4, 6, 9, 10, 11, 13, 15], [-1, 1, 3, 5, 7, 4898])

    def test_singleton(self):
        self._test_containment([111], [110, 112])

    def test_empty(self):
        self._test_containment([], range(100))