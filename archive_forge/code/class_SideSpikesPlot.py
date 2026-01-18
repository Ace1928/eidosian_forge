from collections import defaultdict
import numpy as np
import param
from bokeh.models import CategoricalColorMapper, CustomJS, FactorRange, Range1d, Whisker
from bokeh.models.tools import BoxSelectTool
from bokeh.transform import jitter
from ...core.data import Dataset
from ...core.dimension import dimension_name
from ...core.util import dimension_sanitizer, isfinite
from ...operation import interpolate_curve
from ...util.transform import dim
from ..mixins import AreaMixin, BarsMixin, SpikesMixin
from ..util import compute_sizes, get_min_distance
from .element import ColorbarPlot, ElementPlot, LegendPlot, OverlayPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import (
from .util import categorize_array
class SideSpikesPlot(SpikesPlot):
    """
    SpikesPlot with useful defaults for plotting adjoined rug plot.
    """
    selected = param.List(default=None, doc='\n        The current selection as a list of integers corresponding\n        to the selected items.')
    xaxis = param.ObjectSelector(default='top-bare', objects=['top', 'bottom', 'bare', 'top-bare', 'bottom-bare', None], doc="\n        Whether and where to display the xaxis, bare options allow suppressing\n        all axis labels including ticks and xlabel. Valid options are 'top',\n        'bottom', 'bare', 'top-bare' and 'bottom-bare'.")
    yaxis = param.ObjectSelector(default='right-bare', objects=['left', 'right', 'bare', 'left-bare', 'right-bare', None], doc="\n        Whether and where to display the yaxis, bare options allow suppressing\n        all axis labels including ticks and ylabel. Valid options are 'left',\n        'right', 'bare' 'left-bare' and 'right-bare'.")
    border = param.Integer(default=5, doc='Default borders on plot')
    height = param.Integer(default=50, doc='Height of plot')
    width = param.Integer(default=50, doc='Width of plot')