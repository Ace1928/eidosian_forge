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
def _adjust_cbar(self, cbar, label, dim):
    noalpha = math.floor(self.style[self.cyclic_index].get('alpha', 1)) == 1
    for lb in ['clabel', 'labels']:
        labelsize = self._fontsize(lb, common=False).get('fontsize')
        if labelsize is not None:
            break
    if cbar.solids and noalpha:
        cbar.solids.set_edgecolor('face')
    cbar.set_label(label, fontsize=labelsize)
    if isinstance(self.cbar_ticks, ticker.Locator):
        cbar.ax.yaxis.set_major_locator(self.cbar_ticks)
    elif self.cbar_ticks == 0:
        cbar.set_ticks([])
    elif isinstance(self.cbar_ticks, int):
        locator = ticker.MaxNLocator(self.cbar_ticks)
        cbar.ax.yaxis.set_major_locator(locator)
    elif isinstance(self.cbar_ticks, list):
        if all((isinstance(t, tuple) for t in self.cbar_ticks)):
            ticks, labels = zip(*self.cbar_ticks)
        else:
            ticks, labels = zip(*[(t, dim.pprint_value(t)) for t in self.cbar_ticks])
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
    for tk in ['cticks', 'ticks']:
        ticksize = self._fontsize(tk, common=False).get('fontsize')
        if ticksize is not None:
            cbar.ax.tick_params(labelsize=ticksize)
            break