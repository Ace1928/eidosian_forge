from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
class AreaToLineAxis1Ragged(_AreaToLineLike):

    def validate(self, in_dshape):
        try:
            from datashader.datatypes import RaggedDtype
        except ImportError:
            RaggedDtype = type(None)
        if not isinstance(in_dshape[str(self.x)], RaggedDtype):
            raise ValueError('x must be a RaggedArray')
        elif not isinstance(in_dshape[str(self.y)], RaggedDtype):
            raise ValueError('y must be a RaggedArray')
        elif not isinstance(in_dshape[str(self.y_stack)], RaggedDtype):
            raise ValueError('y_stack must be a RaggedArray')

    def required_columns(self):
        return (self.x, self.y, self.y_stack)

    def compute_x_bounds(self, df):
        bounds = self._compute_bounds(df[self.x].array.flat_array)
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds_y = self._compute_bounds(df[self.y].array.flat_array)
        bounds_y_stack = self._compute_bounds(df[self.y_stack].array.flat_array)
        bounds = (min(bounds_y[0], bounds_y_stack[0], 0), max(bounds_y[1], bounds_y_stack[1], 0))
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[np.nanmin(df[self.x].array.flat_array).item(), np.nanmax(df[self.x].array.flat_array).item(), np.nanmin(df[self.y].array.flat_array).item(), np.nanmax(df[self.y].array.flat_array).item(), np.nanmin(df[self.y_stack].array.flat_array).item(), np.nanmax(df[self.y_stack].array.flat_array).item()]])).compute()
        x_extents = (np.nanmin(r[:, 0]), np.nanmax(r[:, 1]))
        y_extents = (np.nanmin(r[:, [2, 4]]), np.nanmax(r[:, [3, 5]]))
        return (self.maybe_expand_bounds(x_extents), self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(append, map_onto_pixel, expand_aggs_and_cols)
        extend_cpu = _build_extend_area_to_line_axis1_ragged(draw_trapezoid_y, expand_aggs_and_cols)
        x_name = self.x
        y_name = self.y
        y_stack_name = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            xs = df[x_name].values
            ys = df[y_name].values
            y_stacks = df[y_stack_name].values
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, y_stacks, *aggs_and_cols)
        return extend