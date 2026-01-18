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
class geo_hex_binning(hex_binning, project_points):
    """
    Applies hex binning by computing aggregates on a hexagonal grid.

    Should not be user facing as the returned element is not directly
    usable.
    """

    def _process(self, element, key=None):
        if isinstance(element, HexTiles):
            element = project_points._process(self, element)
        return hex_binning._process(self, element)