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
class SideAreaPlot(AdjoinedPlot, AreaPlot):
    bgcolor = param.Parameter(default=(1, 1, 1, 0), doc='\n        Make plot background invisible.')
    border_size = param.Number(default=0, doc='\n        The size of the border expressed as a fraction of the main plot.')
    xaxis = param.ObjectSelector(default='bare', objects=['top', 'bottom', 'bare', 'top-bare', 'bottom-bare', None], doc="\n        Whether and where to display the xaxis, bare options allow suppressing\n        all axis labels including ticks and xlabel. Valid options are 'top',\n        'bottom', 'bare', 'top-bare' and 'bottom-bare'.")
    yaxis = param.ObjectSelector(default='bare', objects=['left', 'right', 'bare', 'left-bare', 'right-bare', None], doc="\n        Whether and where to display the yaxis, bare options allow suppressing\n        all axis labels including ticks and ylabel. Valid options are 'left',\n        'right', 'bare' 'left-bare' and 'right-bare'.")