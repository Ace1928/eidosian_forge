import math
from collections.abc import Sequence
from pyomo.common.numeric_types import check_if_numeric_type
def _step_lcm(self, other_ranges):
    """This computes an approximate Least Common Multiple step"""
    if self.isdiscrete():
        a = self.step or 1
    else:
        a = 0
    for o in other_ranges:
        if o.isdiscrete():
            b = o.step or 1
        else:
            b = 0
        lcm = NumericRange._lcm(a, b)
        if lcm:
            a = lcm
        else:
            a += b
    return int(abs(a))