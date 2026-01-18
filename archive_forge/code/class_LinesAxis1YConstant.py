from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
class LinesAxis1YConstant(LinesAxis1):
    """
    """

    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(xcol)]) for xcol in self.x]):
            raise ValueError('x columns must be real')
        unique_x_measures = set((in_dshape.measure[str(xcol)] for xcol in self.x))
        if len(unique_x_measures) > 1:
            raise ValueError('x columns must have the same data type')
        if len(self.x) != len(self.y):
            raise ValueError(f'x and y coordinate lengths do not match: {len(self.x)} != {len(self.y)}')

    def required_columns(self):
        return self.x

    def compute_y_bounds(self, *args):
        y_min = np.nanmin(self.y)
        y_max = np.nanmax(self.y)
        return self.maybe_expand_bounds((y_min, y_max))

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[np.nanmin([np.nanmin(df[c].values).item() for c in self.x]), np.nanmax([np.nanmax(df[c].values).item() for c in self.x])]])).compute()
        x_extents = (np.nanmin(r[:, 0]), np.nanmax(r[:, 1]))
        return (self.maybe_expand_bounds(x_extents), self.compute_y_bounds())

    @memoize
    def _internal_build_extend(self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_segment, antialias_stage_2_funcs = _line_internal_build_extend(x_mapper, y_mapper, append, line_width, antialias_stage_2, antialias_stage_2_funcs, expand_aggs_and_cols)
        extend_cpu, extend_cuda = _build_extend_line_axis1_y_constant(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs)
        x_names = self.x
        y_values = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, x_names)
                ys = cp.asarray(y_values)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys = y_values
                do_extend = extend_cpu
            do_extend(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols)
        return extend