import copy
import numpy as np
import param
import matplotlib.ticker as mticker
from cartopy import crs as ccrs
from cartopy.io.img_tiles import GoogleTiles, QuadtreeTiles
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from holoviews.core import Store, HoloMap, Layout, Overlay, Element, NdLayout
from holoviews.core import util
from holoviews.core.data import GridInterface
from holoviews.core.options import SkipRendering, Options
from holoviews.plotting.mpl import (
from holoviews.plotting.mpl.util import get_raster_array, wrap_formatter
from ...element import (
from ...util import geo_mesh, poly_types
from ..plot import ProjectionPlot
from ...operation import (
from .chart import WindBarbsPlot
class WMTSPlot(GeoPlot):
    """
    Adds a Web Map Tile Service from a WMTS Element.
    """
    zoom = param.Integer(default=3, doc='\n        Controls the zoom level of the tile source.')
    style_opts = ['alpha', 'cmap', 'interpolation', 'visible', 'filterrad', 'clims', 'norm']

    def get_data(self, element, ranges, style):
        if isinstance(element.data, str):
            if '{Q}' in element.data:
                tile_source = QuadtreeTiles(url=element.data)
            else:
                tile_source = GoogleTiles(url=element.data)
            return ((tile_source, self.zoom), style, {})
        else:
            tile_source = element.data
            return ((tile_source, element.layer), style, {})

    def init_artists(self, ax, plot_args, plot_kwargs):
        if isinstance(plot_args[0], GoogleTiles):
            if 'artist' in self.handles:
                return {'artist': self.handles['artist']}
            img = ax.add_image(*plot_args, **plot_kwargs)
            return {'artist': img or plot_args[0]}
        return {'artist': ax.add_wmts(*plot_args, **plot_kwargs)}

    def teardown_handles(self):
        """
        If no custom update_handles method is supplied this method
        is called to tear down any previous handles before replacing
        them.
        """
        if not isinstance(self.handles.get('artist'), GoogleTiles):
            self.handles['artist'].remove()