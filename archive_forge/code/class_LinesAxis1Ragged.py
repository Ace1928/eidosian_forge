from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
class LinesAxis1Ragged(_PointLike, _AntiAliasedLine):

    def validate(self, in_dshape):
        try:
            from datashader.datatypes import RaggedDtype
        except ImportError:
            RaggedDtype = type(None)
        if not isinstance(in_dshape[str(self.x)], RaggedDtype):
            raise ValueError('x must be a RaggedArray')
        elif not isinstance(in_dshape[str(self.y)], RaggedDtype):
            raise ValueError('y must be a RaggedArray')

    def required_columns(self):
        return (self.x,) + (self.y,)

    def compute_x_bounds(self, df):
        bounds = self._compute_bounds(df[self.x].array.flat_array)
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds = self._compute_bounds(df[self.y].array.flat_array)
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[np.nanmin(df[self.x].array.flat_array).item(), np.nanmax(df[self.x].array.flat_array).item(), np.nanmin(df[self.y].array.flat_array).item(), np.nanmax(df[self.y].array.flat_array).item()]])).compute()
        x_extents = (np.nanmin(r[:, 0]), np.nanmax(r[:, 1]))
        y_extents = (np.nanmin(r[:, 2]), np.nanmax(r[:, 3]))
        return (self.maybe_expand_bounds(x_extents), self.maybe_expand_bounds(y_extents))

    @memoize
    def _internal_build_extend(self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_segment, antialias_stage_2_funcs = _line_internal_build_extend(x_mapper, y_mapper, append, line_width, antialias_stage_2, antialias_stage_2_funcs, expand_aggs_and_cols)
        extend_cpu = _build_extend_line_axis1_ragged(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs)
        x_name = self.x
        y_name = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            xs = df[x_name].array
            ys = df[y_name].array
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, *aggs_and_cols)
        return extend