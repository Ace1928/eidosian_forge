from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
class LineAxis0Multi(_PointLike, _AntiAliasedLine):
    """
    """

    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(xcol)]) for xcol in self.x]):
            raise ValueError('x columns must be real')
        elif not all([isreal(in_dshape.measure[str(ycol)]) for ycol in self.y]):
            raise ValueError('y columns must be real')
        if len(self.x) != len(self.y):
            raise ValueError(f'x and y coordinate lengths do not match: {len(self.x)} != {len(self.y)}')

    @property
    def x_label(self):
        return 'x'

    @property
    def y_label(self):
        return 'y'

    def required_columns(self):
        return self.x + self.y

    def compute_x_bounds(self, df):
        bounds_list = [self._compute_bounds(df[x]) for x in self.x]
        mins, maxes = zip(*bounds_list)
        return self.maybe_expand_bounds((min(mins), max(maxes)))

    def compute_y_bounds(self, df):
        bounds_list = [self._compute_bounds(df[y]) for y in self.y]
        mins, maxes = zip(*bounds_list)
        return self.maybe_expand_bounds((min(mins), max(maxes)))

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[np.nanmin([np.nanmin(df[c].values).item() for c in self.x]), np.nanmax([np.nanmax(df[c].values).item() for c in self.x]), np.nanmin([np.nanmin(df[c].values).item() for c in self.y]), np.nanmax([np.nanmax(df[c].values).item() for c in self.y])]])).compute()
        x_extents = (np.nanmin(r[:, 0]), np.nanmax(r[:, 1]))
        y_extents = (np.nanmin(r[:, 2]), np.nanmax(r[:, 3]))
        return (self.maybe_expand_bounds(x_extents), self.maybe_expand_bounds(y_extents))

    @memoize
    def _internal_build_extend(self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_segment, antialias_stage_2_funcs = _line_internal_build_extend(x_mapper, y_mapper, append, line_width, antialias_stage_2, antialias_stage_2_funcs, expand_aggs_and_cols)
        extend_cpu, extend_cuda = _build_extend_line_axis0_multi(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs)
        x_names = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, x_names)
                ys = self.to_cupy_array(df, y_names)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys = df.loc[:, list(y_names)].to_numpy()
                do_extend = extend_cpu
            do_extend(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, plot_start, antialias_stage_2, *aggs_and_cols)
        return extend