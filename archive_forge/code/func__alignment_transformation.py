from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def _alignment_transformation(segment):
    start = segment[0]
    end = segment[-1]
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    return Identity.rotate(-angle).translate(-start[0], -start[1])