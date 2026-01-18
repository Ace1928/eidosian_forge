from math import copysign, cos, hypot, isclose, pi
from fontTools.misc.roundTools import otRound
def _vector_between(origin, target):
    return (target[0] - origin[0], target[1] - origin[1])