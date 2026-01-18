import unittest
from idna.intranges import intranges_from_list, intranges_contain, _encode_range
def _test_containment(self, ints, disjoint_ints):
    ranges = intranges_from_list(ints)
    for int_ in ints:
        assert intranges_contain(int_, ranges)
    for int_ in disjoint_ints:
        assert not intranges_contain(int_, ranges)