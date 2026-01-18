from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
class LinesAxis1XConstant(LinesAxis1):
    """
    """

    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(ycol)]) for ycol in self.y]):
            raise ValueError('y columns must be real')
        unique_y_measures = set((in_dshape.measure[str(ycol)] for ycol in self.y))
        if len(unique_y_measures) > 1:
            raise ValueError('y columns must have the same data type')
        if len(self.x) != len(self.y):
            raise ValueError(f'x and y coordinate lengths do not match: {len(self.x)} != {len(self.y)}')

    def required_columns(self):
        return self.y

    def compute_x_bounds(self, *args):
        x_min = np.nanmin(self.x)
        x_max = np.nanmax(self.x)
        return self.maybe_expand_bounds((x_min, x_max))

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[np.nanmin([np.nanmin(df[c].values).item() for c in self.y]), np.nanmax([np.nanmax(df[c].values).item() for c in self.y])]])).compute()
        y_extents = (np.nanmin(r[:, 0]), np.nanmax(r[:, 1]))
        return (self.compute_x_bounds(), self.maybe_expand_bounds(y_extents))

    @memoize
    def _internal_build_extend(self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_segment, antialias_stage_2_funcs = _line_internal_build_extend(x_mapper, y_mapper, append, line_width, antialias_stage_2, antialias_stage_2_funcs, expand_aggs_and_cols)
        extend_cpu, extend_cuda = _build_extend_line_axis1_x_constant(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs)
        x_values = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            if cudf and isinstance(df, cudf.DataFrame):
                xs = cp.asarray(x_values)
                ys = self.to_cupy_array(df, y_names)
                do_extend = extend_cuda[cuda_args(ys.shape)]
            else:
                xs = x_values
                ys = df.loc[:, list(y_names)].to_numpy()
                do_extend = extend_cpu
            do_extend(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols)
        return extend