import numpy as np
import param
from bokeh.models.glyphs import AnnularWedge
from ...core.data import GridInterface
from ...core.spaces import HoloMap
from ...core.util import dimension_sanitizer, is_nan
from .element import ColorbarPlot, CompositeElementPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, text_properties
def _draw_markers(self, plot, element, marks, axis='x'):
    if marks is None or self.radial:
        return
    self.param.warning('Only radial HeatMaps supports marks, to make theHeatMap quads for distinguishable set a line_width')