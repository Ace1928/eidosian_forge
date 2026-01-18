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
def _update_glyph(self, renderer, properties, mapping, glyph, source=None, data=None):
    glyph.url = mapping['tile_source'].url
    glyph.update(**{k: v for k, v in properties.items() if k in glyph.properties()})
    renderer.update(**{k: v for k, v in properties.items() if k in renderer.properties()})