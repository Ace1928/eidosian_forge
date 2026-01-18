from toolz import memoize
import numpy as np
from datashader.glyphs.line import _build_map_onto_pixel_for_line
from datashader.glyphs.points import _GeometryLike
from datashader.utils import ngjit
class GeopandasPolygonGeom(_GeometryLike):

    @property
    def geom_dtypes(self):
        from geopandas.array import GeometryDtype
        return (GeometryDtype,)

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_polygon = _build_draw_polygon(append, map_onto_pixel, x_mapper, y_mapper, expand_aggs_and_cols)
        perform_extend_cpu = _build_extend_geopandas_polygon_geometry(draw_polygon, expand_aggs_and_cols)
        geom_name = self.geometry

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            geom_array = df[geom_name].array
            perform_extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, geom_array, *aggs_and_cols)
        return extend