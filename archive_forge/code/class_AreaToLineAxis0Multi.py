from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
class AreaToLineAxis0Multi(_AreaToLineLike):

    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(xcol)]) for xcol in self.x]):
            raise ValueError('x columns must be real')
        elif not all([isreal(in_dshape.measure[str(ycol)]) for ycol in self.y]):
            raise ValueError('y columns must be real')
        elif not all([isreal(in_dshape.measure[str(ycol)]) for ycol in self.y_stack]):
            raise ValueError('y_stack columns must be real')

    @property
    def x_label(self):
        return 'x'

    @property
    def y_label(self):
        return 'y'

    def required_columns(self):
        return self.x + self.y + self.y_stack

    def compute_x_bounds(self, df):
        bounds_list = [self._compute_bounds(df[x]) for x in self.x]
        mins, maxes = zip(*bounds_list)
        return self.maybe_expand_bounds((min(mins), max(maxes)))

    def compute_y_bounds(self, df):
        bounds_list = [self._compute_bounds(df[y]) for y in self.y + self.y_stack]
        mins, maxes = zip(*bounds_list)
        return self.maybe_expand_bounds((min(mins), max(maxes)))

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[np.nanmin([np.nanmin(df[c].values).item() for c in self.x]), np.nanmax([np.nanmax(df[c].values).item() for c in self.x]), np.nanmin([np.nanmin(df[c].values).item() for c in self.y]), np.nanmax([np.nanmax(df[c].values).item() for c in self.y]), np.nanmin([np.nanmin(df[c].values).item() for c in self.y_stack]), np.nanmax([np.nanmax(df[c].values).item() for c in self.y_stack])]])).compute()
        x_extents = (np.nanmin(r[:, 0]), np.nanmax(r[:, 1]))
        y_extents = (np.nanmin(r[:, [2, 4]]), np.nanmax(r[:, [3, 5]]))
        return (self.maybe_expand_bounds(x_extents), self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(append, map_onto_pixel, expand_aggs_and_cols)
        extend_cpu, extend_cuda = _build_extend_area_to_line_axis0_multi(draw_trapezoid_y, expand_aggs_and_cols)
        x_names = self.x
        y_names = self.y
        y_stack_names = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, x_names)
                ys0 = self.to_cupy_array(df, y_names)
                ys1 = self.to_cupy_array(df, y_stack_names)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys0 = df.loc[:, list(y_names)].to_numpy()
                ys1 = df.loc[:, list(y_stack_names)].to_numpy()
                do_extend = extend_cpu
            do_extend(sx, tx, sy, ty, xmin, xmax, ymin, ymax, plot_start, xs, ys0, ys1, *aggs_and_cols)
        return extend