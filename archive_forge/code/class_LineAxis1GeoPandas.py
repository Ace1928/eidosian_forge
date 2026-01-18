from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
class LineAxis1GeoPandas(_GeometryLike, _AntiAliasedLine):

    @property
    def geom_dtypes(self):
        from geopandas.array import GeometryDtype
        return (GeometryDtype,)

    @memoize
    def _internal_build_extend(self, x_mapper, y_mapper, info, append, line_width, antialias_stage_2, antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        draw_segment, antialias_stage_2_funcs = _line_internal_build_extend(x_mapper, y_mapper, append, line_width, antialias_stage_2, antialias_stage_2_funcs, expand_aggs_and_cols)
        perform_extend_cpu = _build_extend_line_axis1_geopandas(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs)
        geometry_name = self.geometry

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            geom_array = df[geometry_name].array
            perform_extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, geom_array, antialias_stage_2, *aggs_and_cols)
        return extend