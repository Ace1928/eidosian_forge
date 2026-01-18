from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
class LinesXarrayCommonX(LinesAxis1):

    def __init__(self, x, y, x_dim_index: int):
        super().__init__(x, y)
        self.x_dim_index = x_dim_index

    def __hash__(self):
        return hash((type(self), self.x_dim_index))

    def compute_x_bounds(self, dataset):
        bounds = self._compute_bounds(dataset[self.x])
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, dataset):
        bounds = self._compute_bounds(dataset[self.y])
        return self.maybe_expand_bounds(bounds)

    def compute_bounds_dask(self, xr_ds):
        return (self.compute_x_bounds(xr_ds), self.compute_y_bounds(xr_ds))

    def validate(self, in_dshape):
        if not isreal(in_dshape.measure[str(self.x)]):
            raise ValueError('x column must be real')
        if not isreal(in_dshape.measure[str(self.y)]):
            raise ValueError('y column must be real')

    @memoize
    def _internal_build_extend(self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_segment, antialias_stage_2_funcs = _line_internal_build_extend(x_mapper, y_mapper, append, line_width, antialias_stage_2, antialias_stage_2_funcs, expand_aggs_and_cols)
        swap_dims = self.x_dim_index == 0
        extend_cpu, extend_cuda = _build_extend_line_axis1_x_constant(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs, swap_dims)
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            if cudf and isinstance(df, cudf.DataFrame):
                xs = cp.asarray(df[x_name])
                ys = cp.asarray(df[y_name])
                do_extend = extend_cuda[cuda_args(ys.shape)]
            elif cp and isinstance(df[y_name].data, cp.ndarray):
                xs = cp.asarray(df[x_name])
                ys = df[y_name].data
                shape = ys.shape[::-1] if swap_dims else ys.shape
                do_extend = extend_cuda[cuda_args(shape)]
            else:
                xs = df[x_name].to_numpy()
                ys = df[y_name].to_numpy()
                do_extend = extend_cpu
            do_extend(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols)
        return extend