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
def _finalize_axis(self, *args, **kwargs):
    gridlabels = self.geographic and isinstance(self.projection, (ccrs.PlateCarree, ccrs.Mercator))
    if gridlabels:
        xaxis, yaxis = (self.xaxis, self.yaxis)
        self.xaxis = self.yaxis = None
    try:
        ret = super()._finalize_axis(*args, **kwargs)
    except Exception as e:
        raise e
    finally:
        if gridlabels:
            self.xaxis, self.yaxis = (xaxis, yaxis)
    axis = self.handles['axis']
    if 'gridlines' in self.handles:
        gl = self.handles['gridlines']
    else:
        self.handles['gridlines'] = gl = axis.gridlines(draw_labels=gridlabels and self.zorder == 0)
    self._process_grid(gl)
    if self.global_extent:
        axis.set_global()
    return ret