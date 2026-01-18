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
def _annotate_plot(self, ax, annotations):
    for a in self.handles.get('annotations', {}).values():
        a.remove()
    handles = {}
    for plot_coord, text in annotations.items():
        handles[plot_coord] = ax.annotate(text, xy=plot_coord, xycoords='data', horizontalalignment='center', verticalalignment='center')
    return handles