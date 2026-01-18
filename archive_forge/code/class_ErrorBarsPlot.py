import numpy as np
import param
from ...element import Tiles
from ...operation import interpolate_curve
from ..mixins import AreaMixin, BarsMixin
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class ErrorBarsPlot(ChartPlot, ColorbarPlot):
    style_opts = ['visible', 'color', 'dash', 'line_width', 'thickness']
    _nonvectorized_styles = style_opts
    _style_key = 'error_y'
    selection_display = PlotlyOverlaySelectionDisplay()

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        return {'type': 'scatter', 'mode': 'lines', 'line': {'width': 0}}

    def get_data(self, element, ranges, style, **kwargs):
        x, y = ('y', 'x') if self.invert_axes else ('x', 'y')
        error_k = 'error_' + x if element.horizontal else 'error_' + y
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)
        error_v = dict(type='data', array=pos_error, arrayminus=neg_error)
        return [{x: element.dimension_values(0), y: element.dimension_values(1), error_k: error_v}]