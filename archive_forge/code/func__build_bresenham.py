from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
def _build_bresenham(expand_aggs_and_cols):
    """Specialize a bresenham kernel for a given append/axis combination"""

    @ngjit
    @expand_aggs_and_cols
    def _bresenham(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, x0, x1, y0, y1, clipped, append, *aggs_and_cols):
        """Draw a line segment using Bresenham's algorithm
        This method plots a line segment with integer coordinates onto a pixel
        grid.
        """
        dx = x1 - x0
        ix = (dx > 0) - (dx < 0)
        dx = abs(dx) * 2
        dy = y1 - y0
        iy = (dy > 0) - (dy < 0)
        dy = abs(dy) * 2
        if not clipped and (not dx | dy):
            append(i, x0, y0, *aggs_and_cols)
            return
        if segment_start:
            append(i, x0, y0, *aggs_and_cols)
        if dx >= dy:
            error = 2 * dy - dx
            while x0 != x1:
                if error >= 0 and (error or ix > 0):
                    error -= 2 * dx
                    y0 += iy
                error += 2 * dy
                x0 += ix
                append(i, x0, y0, *aggs_and_cols)
        else:
            error = 2 * dx - dy
            while y0 != y1:
                if error >= 0 and (error or iy > 0):
                    error -= 2 * dy
                    x0 += ix
                error += 2 * dx
                y0 += iy
                append(i, x0, y0, *aggs_and_cols)
    return _bresenham