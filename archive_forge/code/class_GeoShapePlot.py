import copy
import param
import numpy as np
from cartopy.crs import GOOGLE_MERCATOR
from bokeh.models import WMTSTileSource, BBoxTileSource, QUADKEYTileSource, SaveTool
from holoviews import Store, Overlay, NdOverlay
from holoviews.core import util
from holoviews.core.options import SkipRendering, Options, Compositor
from holoviews.plotting.bokeh.annotation import TextPlot, LabelsPlot
from holoviews.plotting.bokeh.chart import PointPlot, VectorFieldPlot
from holoviews.plotting.bokeh.geometry import RectanglesPlot, SegmentPlot
from holoviews.plotting.bokeh.graphs import TriMeshPlot, GraphPlot
from holoviews.plotting.bokeh.hex_tiles import hex_binning, HexTilesPlot
from holoviews.plotting.bokeh.path import PolygonPlot, PathPlot, ContourPlot
from holoviews.plotting.bokeh.raster import RasterPlot, RGBPlot, QuadMeshPlot
from ...element import (
from ...operation import (
from ...tile_sources import _ATTRIBUTIONS
from ...util import poly_types, line_types
from .plot import GeoPlot, GeoOverlayPlot
from . import callbacks # noqa
class GeoShapePlot(GeoPolygonPlot):

    def get_data(self, element, ranges, style):
        if not isinstance(element.data['geometry'], poly_types):
            style['fill_alpha'] = 0
        if isinstance(element.data['geometry'], line_types):
            el_type = Contours
            style['plot_method'] = 'multi_line'
            style.pop('fill_color', None)
            style.pop('fill_alpha', None)
        else:
            el_type = Polygons
        polys = el_type([element.data], crs=element.crs, **util.get_param_values(element))
        return super().get_data(polys, ranges, style)