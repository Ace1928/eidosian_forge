import numpy as np
import param
from ...element import HLine, HSpan, Tiles, VLine, VSpan
from ..mixins import GeomMixin
from .element import ElementPlot
class PathShapePlot(ShapePlot):
    _shape_type = 'path'

    def get_data(self, element, ranges, style, is_geo=False, **kwargs):
        if self.invert_axes:
            ys = element.dimension_values(0)
            xs = element.dimension_values(1)
        else:
            xs = element.dimension_values(0)
            ys = element.dimension_values(1)
        if is_geo:
            lon, lat = Tiles.easting_northing_to_lon_lat(easting=xs, northing=ys)
            return [{'lat': lat, 'lon': lon}]
        else:
            path = ShapePlot.build_path(xs, ys)
            return [dict(path=path, xref='x', yref='y')]