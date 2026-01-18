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
def _split_area(self, xs, lower, upper):
    """
        Splits area plots at nans and returns x- and y-coordinates for
        each area separated by nans.
        """
    xnan = np.array([np.datetime64('nat') if xs.dtype.kind == 'M' else np.nan])
    ynan = np.array([np.datetime64('nat') if lower.dtype.kind == 'M' else np.nan])
    split = np.where(~isfinite(xs) | ~isfinite(lower) | ~isfinite(upper))[0]
    xvals = np.split(xs, split)
    lower = np.split(lower, split)
    upper = np.split(upper, split)
    band_x, band_y = ([], [])
    for i, (x, l, u) in enumerate(zip(xvals, lower, upper)):
        if i:
            x, l, u = (x[1:], l[1:], u[1:])
        if not len(x):
            continue
        band_x += [np.append(x, x[::-1]), xnan]
        band_y += [np.append(l, u[::-1]), ynan]
    if len(band_x):
        xs = np.concatenate(band_x[:-1])
        ys = np.concatenate(band_y[:-1])
        return (xs, ys)
    return ([], [])