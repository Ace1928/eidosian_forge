from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
class LineAxis1Geometry(_GeometryLike, _AntiAliasedLine):

    @property
    def geom_dtypes(self):
        from spatialpandas.geometry import LineDtype, MultiLineDtype, RingDtype, PolygonDtype, MultiPolygonDtype
        return (LineDtype, MultiLineDtype, RingDtype, PolygonDtype, MultiPolygonDtype)

    @memoize
    def _internal_build_extend(self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        from spatialpandas.geometry import PolygonArray, MultiPolygonArray, RingArray
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_segment, antialias_stage_2_funcs = _line_internal_build_extend(x_mapper, y_mapper, append, line_width, antialias_stage_2, antialias_stage_2_funcs, expand_aggs_and_cols)
        perform_extend_cpu = _build_extend_line_axis1_geometry(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs)
        geometry_name = self.geometry

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            geom_array = df[geometry_name].array
            if isinstance(geom_array, (PolygonArray, MultiPolygonArray)):
                geom_array = geom_array.boundary
                closed_rings = True
            elif isinstance(geom_array, RingArray):
                closed_rings = True
            else:
                closed_rings = False
            perform_extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, geom_array, closed_rings, antialias_stage_2, *aggs_and_cols)
        return extend