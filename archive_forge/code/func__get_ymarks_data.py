import numpy as np
import param
from bokeh.models.glyphs import AnnularWedge
from ...core.data import GridInterface
from ...core.spaces import HoloMap
from ...core.util import dimension_sanitizer, is_nan
from .element import ColorbarPlot, CompositeElementPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, text_properties
def _get_ymarks_data(self, order_ann, bins_ann):
    """
        Generate ColumnDataSource dictionary for segment separation lines.
        """
    if not self.ymarks:
        return dict(radius=[])
    radius = self._get_markers(self.ymarks, order_ann, bins_ann)
    return dict(radius=radius)