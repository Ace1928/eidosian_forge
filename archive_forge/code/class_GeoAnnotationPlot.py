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
class GeoAnnotationPlot(AnnotationPlot):
    """
    AnnotationPlot handles the display of all annotation elements.
    """

    def initialize_plot(self, ranges=None):
        annotation = self.hmap.last
        key = self.keys[-1]
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(annotation, ranges)
        axis = self.handles['axis']
        opts = self.style[self.cyclic_index]
        handles = self.draw_annotation(axis, annotation.data, annotation.crs, opts)
        self.handles['annotations'] = handles
        return self._finalize_axis(key, ranges=ranges)

    def update_handles(self, key, axis, annotation, ranges, style):
        for element in self.handles['annotations']:
            element.remove()
        self.handles['annotations'] = self.draw_annotation(axis, annotation.data, annotation.crs, style)