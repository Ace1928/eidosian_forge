import numpy as np
import param
from ...element import HLine, HSpan, Tiles, VLine, VSpan
from ..mixins import GeomMixin
from .element import ElementPlot
class PathsPlot(ShapePlot):
    _shape_type = 'path'

    def get_data(self, element, ranges, style, is_geo=False, **kwargs):
        if is_geo:
            lon_chunks = []
            lat_chunks = []
            for el in element.split():
                xdim, ydim = (1, 0) if self.invert_axes else (0, 1)
                xs = el.dimension_values(xdim)
                ys = el.dimension_values(ydim)
                el_lon, el_lat = Tiles.easting_northing_to_lon_lat(xs, ys)
                lon_chunks.extend([el_lon, [np.nan]])
                lat_chunks.extend([el_lat, [np.nan]])
            if lon_chunks:
                lon = np.concatenate(lon_chunks)
                lat = np.concatenate(lat_chunks)
            else:
                lon = []
                lat = []
            return [{'lat': lat, 'lon': lon}]
        else:
            paths = []
            for el in element.split():
                xdim, ydim = (1, 0) if self.invert_axes else (0, 1)
                xs = el.dimension_values(xdim)
                ys = el.dimension_values(ydim)
                path = ShapePlot.build_path(xs, ys)
                paths.append(dict(path=path, xref='x', yref='y'))
            return paths