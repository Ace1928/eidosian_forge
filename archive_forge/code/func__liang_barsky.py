from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
@ngjit
def _liang_barsky(xmin, xmax, ymin, ymax, x0, x1, y0, y1, skip):
    """ An implementation of the Liang-Barsky line clipping algorithm.

    https://en.wikipedia.org/wiki/Liang%E2%80%93Barsky_algorithm

    """
    if x0 < xmin and x1 < xmin:
        skip = True
    elif x0 > xmax and x1 > xmax:
        skip = True
    elif y0 < ymin and y1 < ymin:
        skip = True
    elif y0 > ymax and y1 > ymax:
        skip = True
    t0, t1 = (0, 1)
    dx1 = x1 - x0
    t0, t1, accept = _clipt(-dx1, x0 - xmin, t0, t1)
    if not accept:
        skip = True
    t0, t1, accept = _clipt(dx1, xmax - x0, t0, t1)
    if not accept:
        skip = True
    dy1 = y1 - y0
    t0, t1, accept = _clipt(-dy1, y0 - ymin, t0, t1)
    if not accept:
        skip = True
    t0, t1, accept = _clipt(dy1, ymax - y0, t0, t1)
    if not accept:
        skip = True
    if t1 < 1:
        clipped_end = True
        x1 = x0 + t1 * dx1
        y1 = y0 + t1 * dy1
    else:
        clipped_end = False
    if t0 > 0:
        clipped_start = True
        x0 = x0 + t0 * dx1
        y0 = y0 + t0 * dy1
    else:
        clipped_start = False
    return (x0, x1, y0, y1, skip, clipped_start, clipped_end)