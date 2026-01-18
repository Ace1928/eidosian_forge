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
def _clipt(p, q, t0, t1):
    accept = True
    if p < 0 and q < 0:
        r = q / p
        if r > t1:
            accept = False
        elif r > t0:
            t0 = r
    elif p > 0 and q < p:
        r = q / p
        if r < t0:
            accept = False
        elif r < t1:
            t1 = r
    elif q < 0:
        accept = False
    return (t0, t1, accept)