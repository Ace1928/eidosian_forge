from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def _splitQuadraticAtT(a, b, c, *ts):
    ts = list(ts)
    segments = []
    ts.insert(0, 0.0)
    ts.append(1.0)
    ax, ay = a
    bx, by = b
    cx, cy = c
    for i in range(len(ts) - 1):
        t1 = ts[i]
        t2 = ts[i + 1]
        delta = t2 - t1
        delta_2 = delta * delta
        a1x = ax * delta_2
        a1y = ay * delta_2
        b1x = (2 * ax * t1 + bx) * delta
        b1y = (2 * ay * t1 + by) * delta
        t1_2 = t1 * t1
        c1x = ax * t1_2 + bx * t1 + cx
        c1y = ay * t1_2 + by * t1 + cy
        pt1, pt2, pt3 = calcQuadraticPoints((a1x, a1y), (b1x, b1y), (c1x, c1y))
        segments.append((pt1, pt2, pt3))
    return segments