import math
from collections.abc import Sequence
from pyomo.common.numeric_types import check_if_numeric_type
def _nooverlap(self, other):
    """Return True if the ranges for self and other are strictly separate"""
    s1, e1, c1 = self.normalize_bounds()
    s2, e2, c2 = other.normalize_bounds()
    if e1 < s2 or e2 < s1 or (e1 == s2 and (not (c1[1] and c2[0]))) or (e2 == s1 and (not (c2[1] and c1[0]))):
        return True
    return False