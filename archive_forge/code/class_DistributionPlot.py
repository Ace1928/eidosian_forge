import param
from ..mixins import MultiDistributionMixin
from .chart import ChartPlot
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class DistributionPlot(ElementPlot):
    bandwidth = param.Number(default=None, doc='\n        The bandwidth of the kernel for the density estimate.')
    cut = param.Number(default=3, doc='\n        Draw the estimate to cut * bw from the extreme data points.')
    filled = param.Boolean(default=True, doc='\n        Whether the bivariate contours should be filled.')
    style_opts = ['visible', 'color', 'dash', 'line_width']
    _style_key = 'line'
    selection_display = PlotlyOverlaySelectionDisplay()

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        return {'type': 'scatter', 'mode': 'lines'}