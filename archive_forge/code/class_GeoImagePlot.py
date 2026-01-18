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
class GeoImagePlot(GeoPlot, RasterPlot):
    """
    Draws a pcolormesh plot from the data in a Image Element.
    """
    style_opts = ['alpha', 'cmap', 'visible', 'filterrad', 'clims', 'norm']

    def get_data(self, element, ranges, style):
        self._norm_kwargs(element, ranges, style, element.vdims[0])
        style.pop('interpolation', None)
        xs, ys, zs = geo_mesh(element)
        xs = GridInterface._infer_interval_breaks(xs)
        ys = GridInterface._infer_interval_breaks(ys)
        if self.geographic:
            style['transform'] = element.crs
        return ((xs, ys, zs), style, {})

    def init_artists(self, ax, plot_args, plot_kwargs):
        artist = ax.pcolormesh(*plot_args, **plot_kwargs)
        return {'artist': artist}

    def update_handles(self, *args):
        """
        Update the elements of the plot.
        """
        return GeoPlot.update_handles(self, *args)