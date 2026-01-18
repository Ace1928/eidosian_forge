from math import copysign, cos, hypot, isclose, pi
from fontTools.misc.roundTools import otRound
def _unit_vector(vec):
    length = hypot(*vec)
    if length == 0:
        return None
    return (vec[0] / length, vec[1] / length)