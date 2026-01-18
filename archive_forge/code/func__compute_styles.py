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
def _compute_styles(self, element, ranges, style):
    cdim = element.get_dimension(self.color_index)
    color = style.pop('color', None)
    cmap = style.get('cmap', None)
    if cdim and (isinstance(color, str) and color in element or isinstance(color, dim)):
        self.param.warning("Cannot declare style mapping for 'color' option and declare a color_index; ignoring the color_index.")
        cdim = None
    if cdim and cmap:
        cs = element.dimension_values(self.color_index)
        if cs.dtype.kind in 'uif':
            style['c'] = cs
        else:
            style['c'] = search_indices(cs, unique_array(cs))
        self._norm_kwargs(element, ranges, style, cdim)
    elif color is not None:
        style['color'] = color
    style['edgecolors'] = style.pop('edgecolors', style.pop('edgecolor', 'none'))
    ms = style.get('s', mpl.rcParams['lines.markersize'])
    sdim = element.get_dimension(self.size_index)
    if sdim and (isinstance(ms, str) and ms in element or isinstance(ms, dim)):
        self.param.warning("Cannot declare style mapping for 's' option and declare a size_index; ignoring the size_index.")
        sdim = None
    if sdim:
        sizes = element.dimension_values(self.size_index)
        sizes = compute_sizes(sizes, self.size_fn, self.scaling_factor, self.scaling_method, ms)
        if sizes is None:
            eltype = type(element).__name__
            self.param.warning(f'{sdim.pprint_label} dimension is not numeric, cannot use to scale {eltype} size.')
        else:
            style['s'] = sizes
    style['edgecolors'] = style.pop('edgecolors', 'none')