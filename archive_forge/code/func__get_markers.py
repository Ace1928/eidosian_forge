import numpy as np
import param
from bokeh.models.glyphs import AnnularWedge
from ...core.data import GridInterface
from ...core.spaces import HoloMap
from ...core.util import dimension_sanitizer, is_nan
from .element import ColorbarPlot, CompositeElementPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, text_properties
@staticmethod
def _get_markers(marks, order, bins):
    """
        Helper function to get marker positions depending on mark type.
        """
    if callable(marks):
        markers = [x for x in order if marks(x)]
    elif isinstance(marks, list):
        markers = [order[x] for x in marks]
    elif isinstance(marks, tuple):
        markers = marks
    else:
        nth_mark = np.ceil(len(order) / marks).astype(int)
        markers = order[::nth_mark]
    return np.array([bins[x][1] for x in markers])