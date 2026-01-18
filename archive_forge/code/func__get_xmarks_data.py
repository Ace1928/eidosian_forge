import numpy as np
import param
from bokeh.models.glyphs import AnnularWedge
from ...core.data import GridInterface
from ...core.spaces import HoloMap
from ...core.util import dimension_sanitizer, is_nan
from .element import ColorbarPlot, CompositeElementPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, text_properties
def _get_xmarks_data(self, order_seg, bins_seg):
    """
        Generate ColumnDataSource dictionary for segment separation lines.
        """
    if not self.xmarks:
        return dict(xs=[], ys=[])
    angles = self._get_markers(self.xmarks, order_seg, bins_seg)
    inner = self.max_radius * self.radius_inner
    outer = self.max_radius
    y_start = np.sin(angles) * inner + self.max_radius
    y_end = np.sin(angles) * outer + self.max_radius
    x_start = np.cos(angles) * inner + self.max_radius
    x_end = np.cos(angles) * outer + self.max_radius
    xs = zip(x_start, x_end)
    ys = zip(y_start, y_end)
    return dict(xs=list(xs), ys=list(ys))