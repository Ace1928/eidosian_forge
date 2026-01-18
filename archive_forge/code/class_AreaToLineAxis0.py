from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
class AreaToLineAxis0(_AreaToLineLike):
    """A filled area glyph
    The area to be filled is the region from the line defined by ``x`` and
    ``y[0]`` and the line defined by ``x`` and ``y[1]``.

    Parameters
    ----------
    x
        Column names for the x and y coordinates of each vertex.
    y
        List or tuple of length two containing the column names of the
        y-coordinates of the two curves that define the area region.
    """

    def compute_y_bounds(self, df):
        bounds0 = self._compute_bounds(df[self.y])
        bounds1 = self._compute_bounds(df[self.y_stack])
        bounds = (min(bounds0[0], bounds1[0]), max(bounds0[1], bounds1[1]))
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[np.nanmin(df[self.x].values).item(), np.nanmax(df[self.x].values).item(), np.nanmin(df[self.y].values).item(), np.nanmax(df[self.y].values).item(), np.nanmin(df[self.y_stack].values).item(), np.nanmax(df[self.y_stack].values).item()]])).compute()
        x_extents = (np.nanmin(r[:, 0]), np.nanmax(r[:, 1]))
        y0_extents = (np.nanmin(r[:, 2]), np.nanmax(r[:, 3]))
        y1_extents = (np.nanmin(r[:, 4]), np.nanmax(r[:, 5]))
        y_extents = (min(y0_extents[0], y1_extents[0]), max(y0_extents[1], y1_extents[1]))
        return (self.maybe_expand_bounds(x_extents), self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(append, map_onto_pixel, expand_aggs_and_cols)
        extend_cpu, extend_cuda = _build_extend_area_to_line_axis0(draw_trapezoid_y, expand_aggs_and_cols)
        x_name = self.x
        y_name = self.y
        y_stack_name = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, x_name)
                ys0 = self.to_cupy_array(df, y_name)
                ys1 = self.to_cupy_array(df, y_stack_name)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df[x_name].values
                ys0 = df[y_name].values
                ys1 = df[y_stack_name].values
                do_extend = extend_cpu
            do_extend(sx, tx, sy, ty, xmin, xmax, ymin, ymax, plot_start, xs, ys0, ys1, *aggs_and_cols)
        return extend