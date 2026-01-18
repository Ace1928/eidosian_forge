import matplotlib as mpl
import numpy as np
import param
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.dates import DateFormatter, date2num
from packaging.version import Version
from ...core.dimension import Dimension, dimension_name
from ...core.options import Store, abbreviated_exception
from ...core.util import (
from ...element import HeatMap, Raster
from ...operation import interpolate_curve
from ...util.transform import dim
from ..mixins import AreaMixin, BarsMixin, SpikesMixin
from ..plot import PlotSelector
from ..util import compute_sizes, get_min_distance, get_sideplot_ranges
from .element import ColorbarPlot, ElementPlot, LegendPlot
from .path import PathPlot
from .plot import AdjoinedPlot, mpl_rc_context
from .util import mpl_version
def _update_artists(self, n, element, edges, hvals, widths, lims, ranges):
    super()._update_artists(n, element, edges, hvals, widths, lims, ranges)
    self._update_plot(n, element, self.handles['artist'], lims, ranges)