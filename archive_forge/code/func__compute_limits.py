import copy
import math
import warnings
from types import FunctionType
import matplotlib.colors as mpl_colors
import numpy as np
import param
from matplotlib import ticker
from matplotlib.dates import date2num
from matplotlib.image import AxesImage
from packaging.version import Version
from ...core import (
from ...core.options import Keywords, abbreviated_exception
from ...element import Graph, Path
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_range_key, process_cmap
from .plot import MPLPlot, mpl_rc_context
from .util import EqHistNormalize, mpl_version, validate, wrap_formatter
def _compute_limits(self, low, high, log, invert, low_key, high_key):
    scale = True
    lims = {}
    valid_lim = lambda c: util.isnumeric(c) and (not np.isnan(c))
    if not isinstance(low, util.datetime_types) and log and (low is None or low <= 0):
        low = 0.01 if high < 0.01 else 10 ** (np.log10(high) - 2)
        self.param.warning('Logarithmic axis range encountered value less than or equal to zero, please supply explicit lower-bound to override default of %.3f.' % low)
    if invert:
        high, low = (low, high)
    if isinstance(low, util.cftime_types) or low != high:
        if valid_lim(low):
            lims[low_key] = low
            scale = False
        if valid_lim(high):
            lims[high_key] = high
            scale = False
    return (lims, scale)