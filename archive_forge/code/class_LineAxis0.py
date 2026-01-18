from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
class LineAxis0(_PointLike, _AntiAliasedLine):
    """A line, with vertices defined by ``x`` and ``y``.

    Parameters
    ----------
    x, y : str
        Column names for the x and y coordinates of each vertex.
    """

    @memoize
    def _internal_build_extend(self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_segment, antialias_stage_2_funcs = _line_internal_build_extend(x_mapper, y_mapper, append, line_width, antialias_stage_2, antialias_stage_2_funcs, expand_aggs_and_cols)
        extend_cpu, extend_cuda = _build_extend_line_axis0(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs)
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, x_name)
                ys = self.to_cupy_array(df, y_name)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, x_name].to_numpy()
                ys = df.loc[:, y_name].to_numpy()
                do_extend = extend_cpu
            do_extend(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, plot_start, antialias_stage_2, *aggs_and_cols)
        return extend