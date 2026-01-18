from __future__ import annotations
from math import floor
import numpy as np
from toolz import memoize
from datashader.glyphs.points import _PointLike
from datashader.utils import isreal, ngjit
class Triangles(_PolygonLike):
    """An unstructured mesh of triangles, with vertices defined by ``xs`` and ``ys``.

    Parameters
    ----------
    xs, ys, zs : list of str
        Column names of x, y, and (optional) z coordinates of each vertex.
    """

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        draw_triangle, draw_triangle_interp = _build_draw_triangle(append)
        map_onto_pixel = _build_map_onto_pixel_for_triangle(x_mapper, y_mapper)
        extend_triangles = _build_extend_triangles(draw_triangle, draw_triangle_interp, map_onto_pixel)
        weight_type = self.weight_type
        interpolate = self.interpolate

        def extend(aggs, df, vt, bounds, plot_start=True):
            cols = info(df, aggs[0].shape[:2])
            assert cols, 'There must be at least one column on which to aggregate'
            extend_triangles(vt, bounds, df.values, weight_type, interpolate, aggs, cols)
        return extend