from fontTools.pens.basePen import BasePen
from fontTools.misc.bezierTools import (
import math
def _distance(p0, p1):
    return math.hypot(p0[0] - p1[0], p0[1] - p1[1])