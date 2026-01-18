import math
from collections.abc import Sequence
from pyomo.common.numeric_types import check_if_numeric_type
@staticmethod
def _continuous_discrete_disjoint(cont, disc):
    d_lb = disc.start if disc.step > 0 else disc.end
    d_ub = disc.end if disc.step > 0 else disc.start
    if cont.start <= d_lb:
        return False
    if cont.end >= d_ub:
        return False
    EPS = NumericRange._EPS
    if cont.end - cont.start - EPS > abs(disc.step):
        return False
    rStart = remainder(cont.start - disc.start, abs(disc.step))
    rEnd = remainder(cont.end - disc.start, abs(disc.step))
    return (abs(rStart) > EPS or not cont.closed[0]) and (abs(rEnd) > EPS or not cont.closed[1]) and (rStart - rEnd > 0 or not any(cont.closed))