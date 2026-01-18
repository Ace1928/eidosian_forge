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
def _x_intercept(y, cx0, cy0, cx1, cy1):
    if cy0 == cy1:
        return cx1
    frac = (y - cy0) / (cy1 - cy0)
    return cx0 + frac * (cx1 - cx0)