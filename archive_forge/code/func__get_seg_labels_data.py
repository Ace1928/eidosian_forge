import numpy as np
import param
from bokeh.models.glyphs import AnnularWedge
from ...core.data import GridInterface
from ...core.spaces import HoloMap
from ...core.util import dimension_sanitizer, is_nan
from .element import ColorbarPlot, CompositeElementPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, text_properties
def _get_seg_labels_data(self, order_seg, bins_seg):
    """
        Generate ColumnDataSource dictionary for segment labels.
        """
    if self.xticks is None:
        return dict(x=[], y=[], text=[], angle=[])
    mapping = self._compute_tick_mapping('angle', order_seg, bins_seg)
    values = [(text, (end - start) / 2 + start) for text, (start, end) in mapping.items()]
    labels, radiant = zip(*values)
    radiant = np.array(radiant)
    y_coord = np.sin(radiant) * self.max_radius + self.max_radius
    x_coord = np.cos(radiant) * self.max_radius + self.max_radius
    return dict(x=x_coord, y=y_coord, text=labels, angle=1.5 * np.pi + radiant)