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
def _set_axis_ticks(self, axis, ticks, log=False, rotation=0):
    """
        Allows setting the ticks for a particular axis either with
        a tuple of ticks, a tick locator object, an integer number
        of ticks, a list of tuples containing positions and labels
        or a list of positions. Also supports enabling log ticking
        if an integer number of ticks is supplied and setting a
        rotation for the ticks.
        """
    if isinstance(ticks, np.ndarray):
        ticks = list(ticks)
    if isinstance(ticks, (list, tuple)) and all((isinstance(l, list) for l in ticks)):
        axis.set_ticks(ticks[0])
        axis.set_ticklabels(ticks[1])
    elif isinstance(ticks, ticker.Locator):
        axis.set_major_locator(ticks)
    elif ticks is not None and (not ticks):
        axis.set_ticks([])
    elif isinstance(ticks, int):
        if log:
            locator = ticker.LogLocator(numticks=ticks, subs=range(1, 10))
        else:
            locator = ticker.MaxNLocator(ticks)
        axis.set_major_locator(locator)
    elif isinstance(ticks, (list, tuple)):
        labels = None
        if all((isinstance(t, tuple) for t in ticks)):
            ticks, labels = zip(*ticks)
        axis.set_ticks(ticks)
        if labels:
            axis.set_ticklabels(labels)
    for tick in axis.get_ticklabels():
        tick.set_rotation(rotation)