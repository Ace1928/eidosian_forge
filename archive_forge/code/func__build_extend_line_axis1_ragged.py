from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
def _build_extend_line_axis1_ragged(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs):
    if antialias_stage_2_funcs is not None:
        aa_stage_2_accumulate, aa_stage_2_clear, aa_stage_2_copy_back = antialias_stage_2_funcs
    use_2_stage_agg = antialias_stage_2_funcs is not None

    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols):
        x_start_i = xs.start_indices
        x_flat = xs.flat_array
        y_start_i = ys.start_indices
        y_flat = ys.flat_array
        extend_cpu_numba(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x_start_i, x_flat, y_start_i, y_flat, antialias_stage_2, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x_start_i, x_flat, y_start_i, y_flat, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        nrows = len(x_start_i)
        x_flat_len = len(x_flat)
        y_flat_len = len(y_flat)
        for i in range(nrows):
            x_start_index = x_start_i[i]
            x_stop_index = x_start_i[i + 1] if i < nrows - 1 else x_flat_len
            y_start_index = y_start_i[i]
            y_stop_index = y_start_i[i + 1] if i < nrows - 1 else y_flat_len
            segment_len = min(x_stop_index - x_start_index, y_stop_index - y_start_index)
            for j in range(segment_len - 1):
                x0 = x_flat[x_start_index + j]
                y0 = y_flat[y_start_index + j]
                x1 = x_flat[x_start_index + j + 1]
                y1 = y_flat[y_start_index + j + 1]
                segment_start = j == 0 or isnull(x_flat[x_start_index + j - 1]) or isnull(y_flat[y_start_index + j - 1])
                segment_end = j == segment_len - 2 or isnull(x_flat[x_start_index + j + 2]) or isnull(y_flat[y_start_index + j + 2])
                if segment_start or use_2_stage_agg:
                    xm = 0.0
                    ym = 0.0
                else:
                    xm = x_flat[x_start_index + j - 1]
                    ym = y_flat[y_start_index + j - 1]
                draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, segment_end, x0, x1, y0, y1, xm, ym, buffer, *aggs_and_cols)

    def extend_cpu_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols):
        x_start_i = xs.start_indices
        x_flat = xs.flat_array
        y_start_i = ys.start_indices
        y_flat = ys.flat_array
        n_aggs = len(antialias_stage_2[0])
        aggs_and_accums = tuple(((agg, agg.copy()) for agg in aggs_and_cols[:n_aggs]))
        extend_cpu_numba_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x_start_i, x_flat, y_start_i, y_flat, antialias_stage_2, aggs_and_accums, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x_start_i, x_flat, y_start_i, y_flat, antialias_stage_2, aggs_and_accums, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        nrows = len(x_start_i)
        x_flat_len = len(x_flat)
        y_flat_len = len(y_flat)
        for i in range(nrows):
            x_start_index = x_start_i[i]
            x_stop_index = x_start_i[i + 1] if i < nrows - 1 else x_flat_len
            y_start_index = y_start_i[i]
            y_stop_index = y_start_i[i + 1] if i < nrows - 1 else y_flat_len
            segment_len = min(x_stop_index - x_start_index, y_stop_index - y_start_index)
            for j in range(segment_len - 1):
                x0 = x_flat[x_start_index + j]
                y0 = y_flat[y_start_index + j]
                x1 = x_flat[x_start_index + j + 1]
                y1 = y_flat[y_start_index + j + 1]
                segment_start = j == 0 or isnull(x_flat[x_start_index + j - 1]) or isnull(y_flat[y_start_index + j - 1])
                segment_end = j == segment_len - 2 or isnull(x_flat[x_start_index + j + 2]) or isnull(y_flat[y_start_index + j + 2])
                draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, segment_end, x0, x1, y0, y1, 0.0, 0.0, buffer, *aggs_and_cols)
            if nrows == 1:
                return
            aa_stage_2_accumulate(aggs_and_accums, i == 0)
            if i < nrows - 1:
                aa_stage_2_clear(aggs_and_accums)
        aa_stage_2_copy_back(aggs_and_accums)
    if use_2_stage_agg:
        return extend_cpu_antialias_2agg
    else:
        return extend_cpu