from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def _split_segment_at_t(c, t):
    if len(c) == 2:
        s, e = c
        midpoint = linePointAtT(s, e, t)
        return [(s, midpoint), (midpoint, e)]
    if len(c) == 3:
        return splitQuadraticAtT(*c, t)
    elif len(c) == 4:
        return splitCubicAtT(*c, t)
    raise ValueError('Unknown curve degree')