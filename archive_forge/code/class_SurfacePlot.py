import numpy as np
import param
from plotly import colors
from plotly.figure_factory._trisurf import trisurf as trisurface
from ...core.options import SkipRendering
from .chart import CurvePlot, ScatterPlot
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class SurfacePlot(Chart3DPlot, ColorbarPlot):
    style_opts = ['visible', 'alpha', 'lighting', 'lightposition', 'cmap']
    selection_display = PlotlyOverlaySelectionDisplay(supports_region=False)

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        return {'type': 'surface'}

    def graph_options(self, element, ranges, style, **kwargs):
        opts = super().graph_options(element, ranges, style, **kwargs)
        copts = self.get_color_opts(element.vdims[0], element, ranges, style)
        return dict(opts, **copts)

    def get_data(self, element, ranges, style, **kwargs):
        return [dict(x=element.dimension_values(0, False), y=element.dimension_values(1, False), z=element.dimension_values(2, flat=False))]