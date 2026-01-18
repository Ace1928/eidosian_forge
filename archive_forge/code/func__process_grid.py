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
def _process_grid(self, gl):
    if not self.show_grid:
        gl.xlines = False
        gl.ylines = False
    if self.xaxis and self.xaxis != 'bare':
        xticksize = self._fontsize('xticks', common=False).get('fontsize')
        gl.xlabel_style = {'size': xticksize}
        if isinstance(self.xticks, list):
            gl.xlocator = mticker.FixedLocator(self.xticks)
        elif isinstance(self.xticks, int):
            gl.xlocator = mticker.MaxNLocator(self.xticks)
        if self.xaxis in ['bottom', 'top-bare']:
            gl.top_labels = False
        elif self.xaxis in ['top', 'bottom-bare']:
            gl.bottom_labels = False
        if self.xformatter is None:
            gl.xformatter = LONGITUDE_FORMATTER
        else:
            gl.xformatter = wrap_formatter(self.xformatter)
    else:
        gl.top_labels = False
        gl.bottom_labels = False
    if self.yaxis and self.yaxis != 'bare':
        yticksize = self._fontsize('yticks', common=False).get('fontsize')
        gl.ylabel_style = {'size': yticksize}
        if isinstance(self.yticks, list):
            gl.ylocator = mticker.FixedLocator(self.yticks)
        elif isinstance(self.yticks, int):
            gl.ylocator = mticker.MaxNLocator(self.yticks)
        if self.yaxis in ['left', 'right-bare']:
            gl.right_labels = False
        elif self.yaxis in ['right', 'left-bare']:
            gl.left_labels = False
        if self.yformatter is None:
            gl.yformatter = LATITUDE_FORMATTER
        else:
            gl.yformatter = wrap_formatter(self.yformatter)
    else:
        gl.left_labels = False
        gl.right_labels = False