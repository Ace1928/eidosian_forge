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
def _finalize_ticks(self, axis, dimensions, xticks, yticks, zticks):
    """
        Finalizes the ticks on the axes based on the supplied ticks
        and Elements. Sets the axes position as well as tick positions,
        labels and fontsize.
        """
    ndims = len(dimensions) if dimensions else 0
    xdim = dimensions[0] if ndims else None
    ydim = dimensions[1] if ndims > 1 else None
    if xdim:
        self._set_axis_formatter(axis.xaxis, xdim, self.xformatter)
    if ydim:
        self._set_axis_formatter(axis.yaxis, ydim, self.yformatter)
    if isinstance(self.projection, str) and self.projection == '3d':
        zdim = dimensions[2] if ndims > 2 else None
        if zdim or self.zformatter is not None:
            self._set_axis_formatter(axis.zaxis, zdim, self.zformatter)
    xticks = xticks if xticks else self.xticks
    self._set_axis_ticks(axis.xaxis, xticks, log=self.logx, rotation=self.xrotation)
    yticks = yticks if yticks else self.yticks
    self._set_axis_ticks(axis.yaxis, yticks, log=self.logy, rotation=self.yrotation)
    if isinstance(self.projection, str) and self.projection == '3d':
        zticks = zticks if zticks else self.zticks
        self._set_axis_ticks(axis.zaxis, zticks, log=self.logz, rotation=self.zrotation)
    axes_str = 'xy'
    axes_list = [axis.xaxis, axis.yaxis]
    if hasattr(axis, 'zaxis'):
        axes_str += 'z'
        axes_list.append(axis.zaxis)
    for ax, ax_obj in zip(axes_str, axes_list):
        tick_fontsize = self._fontsize(f'{ax}ticks', 'labelsize', common=False)
        if tick_fontsize:
            ax_obj.set_tick_params(**tick_fontsize)