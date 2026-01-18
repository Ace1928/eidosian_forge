import sys
import numpy as np
import param
from packaging.version import Version
from ...core import CompositeOverlay, Element, traversal
from ...core.util import isfinite, match_spec, max_range, unique_iterator
from ...element.raster import RGB, Image, Raster
from ..util import categorical_legend
from .chart import PointPlot
from .element import ColorbarPlot, ElementPlot, LegendPlot, OverlayPlot
from .plot import GridPlot, MPLPlot, mpl_rc_context
from .util import get_raster_array, mpl_version
def _compute_borders(self):
    ndims = self.layout.ndims
    width_fn = lambda x: x.range(0)
    height_fn = lambda x: x.range(1)
    width_extents = [max_range(self.layout[x, :].traverse(width_fn, [Element])) for x in unique_iterator(self.layout.dimension_values(0))]
    if ndims > 1:
        height_extents = [max_range(self.layout[:, y].traverse(height_fn, [Element])) for y in unique_iterator(self.layout.dimension_values(1))]
    else:
        height_extents = [max_range(self.layout.traverse(height_fn, [Element]))]
    widths = [extent[0] - extent[1] for extent in width_extents]
    heights = [extent[0] - extent[1] for extent in height_extents]
    width, height = (np.sum(widths), np.sum(heights))
    border_width = width * self.padding / (len(widths) + 1)
    border_height = height * self.padding / (len(heights) + 1)
    width += width * self.padding
    height += height * self.padding
    return (width, height, border_width, border_height, widths, heights)