from itertools import product
import numpy as np
import param
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Wedge
from ...core.data import GridInterface
from ...core.spaces import HoloMap
from ...core.util import dimension_sanitizer, is_nan
from ..mixins import HeatMapMixin
from .element import ColorbarPlot
from .raster import QuadMeshPlot
from .util import filter_styles
def _annotate_values(self, element, xvals, yvals):
    val_dim = element.vdims[0]
    vals = element.dimension_values(val_dim).flatten()
    xpos = xvals[:-1] + np.diff(xvals) / 2.0
    ypos = yvals[:-1] + np.diff(yvals) / 2.0
    plot_coords = product(xpos, ypos)
    annotations = {}
    for plot_coord, v in zip(plot_coords, vals):
        text = '-' if is_nan(v) else val_dim.pprint_value(v)
        annotations[plot_coord] = text
    return annotations