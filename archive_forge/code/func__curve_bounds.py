from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def _curve_bounds(c):
    if len(c) == 3:
        return calcQuadraticBounds(*c)
    elif len(c) == 4:
        return calcCubicBounds(*c)
    raise ValueError('Unknown curve degree')