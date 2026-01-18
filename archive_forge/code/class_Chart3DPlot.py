import numpy as np
import param
from plotly import colors
from plotly.figure_factory._trisurf import trisurf as trisurface
from ...core.options import SkipRendering
from .chart import CurvePlot, ScatterPlot
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class Chart3DPlot(ElementPlot):
    aspect = param.Parameter(default='cube')
    camera_angle = param.NumericTuple(default=(0.2, 0.5, 0.1, 0.2))
    camera_position = param.NumericTuple(default=(0.1, 0, -0.1))
    camera_zoom = param.Integer(default=3)
    projection = param.String(default='3d')
    width = param.Integer(default=500)
    height = param.Integer(default=500)
    zticks = param.Parameter(default=None, doc='\n        Ticks along z-axis specified as an integer, explicit list of\n        tick locations, list of tuples containing the locations.')

    def get_data(self, element, ranges, style, **kwargs):
        return [dict(x=element.dimension_values(0), y=element.dimension_values(1), z=element.dimension_values(2))]