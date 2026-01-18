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
@staticmethod
def _map_order_to_ticks(start, end, order, reverse=False):
    """Map elements from given `order` array to bins ranging from `start`
        to `end`.
        """
    size = len(order)
    bounds = np.linspace(start, end, size + 1)
    if reverse:
        bounds = bounds[::-1]
    mapping = list(zip(bounds[:-1] % (np.pi * 2), order))
    return mapping