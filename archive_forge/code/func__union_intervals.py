from ..sage_helper import _within_sage
from ..math_basics import correct_min, is_RealIntervalFieldElement
def _union_intervals(intervals):
    result = intervals[0]
    for i in intervals[1:]:
        result = result.union(i)
    return result