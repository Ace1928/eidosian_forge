from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
def _build_extend_line_axis1_x_constant(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs, swap_dims: bool=False):
    if antialias_stage_2_funcs is not None:
        aa_stage_2_accumulate, aa_stage_2_clear, aa_stage_2_copy_back = antialias_stage_2_funcs
    use_2_stage_agg = antialias_stage_2_funcs is not None

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, buffer, *aggs_and_cols):
        x0 = xs[j]
        x1 = xs[j + 1]
        if swap_dims:
            y0 = ys[j, i]
            y1 = ys[j + 1, i]
            segment_start = j == 0 or isnull(xs[j - 1]) or isnull(ys[j - 1, i])
            segment_end = j == len(xs) - 2 or isnull(xs[j + 2]) or isnull(ys[j + 2, i])
        else:
            y0 = ys[i, j]
            y1 = ys[i, j + 1]
            segment_start = j == 0 or isnull(xs[j - 1]) or isnull(ys[i, j - 1])
            segment_end = j == len(xs) - 2 or isnull(xs[j + 2]) or isnull(ys[i, j + 2])
        if segment_start or use_2_stage_agg:
            xm = 0.0
            ym = 0.0
        else:
            xm = xs[j - 1]
            ym = ys[j - 1, i] if swap_dims else ys[i, j - 1]
        draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, segment_end, x0, x1, y0, y1, xm, ym, buffer, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        ncols, nrows = ys.shape if swap_dims else ys.shape[::-1]
        for i in range(nrows):
            for j in range(ncols - 1):
                perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, buffer, *aggs_and_cols)

    def extend_cpu_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols):
        n_aggs = len(antialias_stage_2[0])
        aggs_and_accums = tuple(((agg, agg.copy()) for agg in aggs_and_cols[:n_aggs]))
        cpu_antialias_2agg_impl(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, aggs_and_accums, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def cpu_antialias_2agg_impl(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, aggs_and_accums, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        ncols = ys.shape[1]
        for i in range(ys.shape[0]):
            for j in range(ncols - 1):
                perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, buffer, *aggs_and_cols)
            if ys.shape[0] == 1:
                return
            aa_stage_2_accumulate(aggs_and_accums, i == 0)
            if i < ys.shape[0] - 1:
                aa_stage_2_clear(aggs_and_accums)
        aa_stage_2_copy_back(aggs_and_accums)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = cuda.local.array(8, nb_types.float64) if antialias else None
        i, j = cuda.grid(2)
        ncols, nrows = ys.shape if swap_dims else ys.shape[::-1]
        if i < nrows and j < ncols - 1:
            perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, buffer, *aggs_and_cols)
    if use_2_stage_agg:
        return (extend_cpu_antialias_2agg, extend_cuda)
    else:
        return (extend_cpu, extend_cuda)