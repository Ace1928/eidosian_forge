from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def _is_linelike(segment):
    maybeline = _alignment_transformation(segment).transformPoints(segment)
    return all((math.isclose(p[1], 0.0) for p in maybeline))