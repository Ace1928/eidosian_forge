from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def _curve_curve_intersections_t(curve1, curve2, precision=0.001, range1=None, range2=None):
    bounds1 = _curve_bounds(curve1)
    bounds2 = _curve_bounds(curve2)
    if not range1:
        range1 = (0.0, 1.0)
    if not range2:
        range2 = (0.0, 1.0)
    intersects, _ = sectRect(bounds1, bounds2)
    if not intersects:
        return []

    def midpoint(r):
        return 0.5 * (r[0] + r[1])
    if rectArea(bounds1) < precision and rectArea(bounds2) < precision:
        return [(midpoint(range1), midpoint(range2))]
    c11, c12 = _split_segment_at_t(curve1, 0.5)
    c11_range = (range1[0], midpoint(range1))
    c12_range = (midpoint(range1), range1[1])
    c21, c22 = _split_segment_at_t(curve2, 0.5)
    c21_range = (range2[0], midpoint(range2))
    c22_range = (midpoint(range2), range2[1])
    found = []
    found.extend(_curve_curve_intersections_t(c11, c21, precision, range1=c11_range, range2=c21_range))
    found.extend(_curve_curve_intersections_t(c12, c21, precision, range1=c12_range, range2=c21_range))
    found.extend(_curve_curve_intersections_t(c11, c22, precision, range1=c11_range, range2=c22_range))
    found.extend(_curve_curve_intersections_t(c12, c22, precision, range1=c12_range, range2=c22_range))
    unique_key = lambda ts: (int(ts[0] / precision), int(ts[1] / precision))
    seen = set()
    unique_values = []
    for ts in found:
        key = unique_key(ts)
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(ts)
    return unique_values