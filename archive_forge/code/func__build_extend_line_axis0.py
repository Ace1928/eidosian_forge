from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
def _build_extend_line_axis0(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs):
    use_2_stage_agg = antialias_stage_2_funcs is not None

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_line(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, plot_start, xs, ys, buffer, *aggs_and_cols):
        x0 = xs[i]
        y0 = ys[i]
        x1 = xs[i + 1]
        y1 = ys[i + 1]
        segment_start = plot_start if i == 0 else isnull(xs[i - 1]) or isnull(ys[i - 1])
        segment_end = i == len(xs) - 2 or isnull(xs[i + 2]) or isnull(ys[i + 2])
        if segment_start or use_2_stage_agg:
            xm = 0.0
            ym = 0.0
        else:
            xm = xs[i - 1]
            ym = ys[i - 1]
        draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, segment_end, x0, x1, y0, y1, xm, ym, buffer, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, plot_start, antialias_stage_2, *aggs_and_cols):
        """Aggregate along a line formed by ``xs`` and ``ys``"""
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        nrows = xs.shape[0]
        for i in range(nrows - 1):
            perform_extend_line(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, plot_start, xs, ys, buffer, *aggs_and_cols)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, plot_start, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = cuda.local.array(8, nb_types.float64) if antialias else None
        i = cuda.grid(1)
        if i < xs.shape[0] - 1:
            perform_extend_line(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, plot_start, xs, ys, buffer, *aggs_and_cols)
    return (extend_cpu, extend_cuda)