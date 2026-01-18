import sys
import numpy as np
import param
from bokeh.models import CustomJSHover, DatetimeAxis
from ...core.util import cartesian_product, dimension_sanitizer, isfinite
from ...element import Raster
from ..util import categorical_legend
from .chart import PointPlot
from .element import ColorbarPlot, LegendPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, mpl_to_bokeh
from .util import bokeh33, bokeh34, colormesh
class HSVPlot(RGBPlot):

    def get_data(self, element, ranges, style):
        return super().get_data(element.rgb, ranges, style)