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
def _linearstep(edge0, edge1, x):
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t