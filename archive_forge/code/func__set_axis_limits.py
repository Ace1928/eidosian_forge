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
def _set_axis_limits(self, axis, view, subplots, ranges):
    """
        Compute extents for current view and apply as axis limits
        """
    extents = self.get_extents(view, ranges)
    if not extents or self.overlaid:
        axis.autoscale_view(scalex=True, scaley=True)
        return
    valid_lim = lambda c: util.isnumeric(c) and (not np.isnan(c))
    coords = [coord if isinstance(coord, np.datetime64) or np.isreal(coord) else np.nan for coord in extents]
    coords = [date2num(util.dt64_to_dt(c)) if isinstance(c, np.datetime64) else c for c in coords]
    if isinstance(self.projection, str) and self.projection == '3d' or len(extents) == 6:
        l, b, zmin, r, t, zmax = coords
        if self.invert_zaxis or any((p.invert_zaxis for p in subplots)):
            zmin, zmax = (zmax, zmin)
        if zmin != zmax:
            if valid_lim(zmin):
                axis.set_zlim(bottom=zmin)
            if valid_lim(zmax):
                axis.set_zlim(top=zmax)
    elif isinstance(self.projection, str) and self.projection == 'polar':
        _, b, _, t = coords
        l = 0
        r = 2 * np.pi
    else:
        l, b, r, t = coords
    if self.invert_axes:
        l, b, r, t = (b, l, t, r)
    invertx = self.invert_xaxis or any((p.invert_xaxis for p in subplots))
    xlim, scalex = self._compute_limits(l, r, self.logx, invertx, 'left', 'right')
    inverty = self.invert_yaxis or any((p.invert_yaxis for p in subplots))
    ylim, scaley = self._compute_limits(b, t, self.logy, inverty, 'bottom', 'top')
    if xlim:
        axis.set_xlim(**xlim)
    if ylim:
        axis.set_ylim(**ylim)
    axis.autoscale_view(scalex=scalex, scaley=scaley)