from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
class AreaToLineAxis1XConstant(AreaToLineAxis1):

    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(ycol)]) for ycol in self.y]):
            raise ValueError('y columns must be real')
        if not all([isreal(in_dshape.measure[str(ycol)]) for ycol in self.y_stack]):
            raise ValueError('y_stack columns must be real')
        unique_y_measures = set((in_dshape.measure[str(ycol)] for ycol in self.y))
        if len(unique_y_measures) > 1:
            raise ValueError('y columns must have the same data type')
        unique_y_stack_measures = set((in_dshape.measure[str(ycol)] for ycol in self.y))
        if len(unique_y_stack_measures) > 1:
            raise ValueError('y_stack columns must have the same data type')

    def required_columns(self):
        return self.y + self.y_stack

    def compute_x_bounds(self, *args):
        x_min = np.nanmin(self.x)
        x_max = np.nanmax(self.x)
        return self.maybe_expand_bounds((x_min, x_max))

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[np.nanmin([np.nanmin(df[c].values).item() for c in self.y]), np.nanmax([np.nanmax(df[c].values).item() for c in self.y]), np.nanmin([np.nanmin(df[c].values).item() for c in self.y_stack]), np.nanmax([np.nanmax(df[c].values).item() for c in self.y_stack])]])).compute()
        y_extents = (np.nanmin(r[:, [0, 2]]), np.nanmax(r[:, [1, 3]]))
        return (self.compute_x_bounds(), self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(append, map_onto_pixel, expand_aggs_and_cols)
        extend_cpu, extend_cuda = _build_extend_area_to_line_axis1_x_constant(draw_trapezoid_y, expand_aggs_and_cols)
        x_values = self.x
        y_names = self.y
        y_stack_names = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            if cudf and isinstance(df, cudf.DataFrame):
                ys = self.to_cupy_array(df, list(y_names))
                y_stacks = self.to_cupy_array(df, list(y_stack_names))
                do_extend = extend_cuda[cuda_args(ys.shape)]
            else:
                ys = df.loc[:, list(y_names)].to_numpy()
                y_stacks = df.loc[:, list(y_stack_names)].to_numpy()
                do_extend = extend_cpu
            do_extend(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x_values, ys, y_stacks, *aggs_and_cols)
        return extend