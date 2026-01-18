import sys
import numpy as np
import param
from packaging.version import Version
from ...core import CompositeOverlay, Element, traversal
from ...core.util import isfinite, match_spec, max_range, unique_iterator
from ...element.raster import RGB, Image, Raster
from ..util import categorical_legend
from .chart import PointPlot
from .element import ColorbarPlot, ElementPlot, LegendPlot, OverlayPlot
from .plot import GridPlot, MPLPlot, mpl_rc_context
from .util import get_raster_array, mpl_version
class RasterBasePlot(ElementPlot):
    aspect = param.Parameter(default='equal', doc="\n        Raster elements respect the aspect ratio of the\n        Images by default but may be set to an explicit\n        aspect ratio or to 'square'.")
    nodata = param.Integer(default=None, doc='\n        Optional missing-data value for integer data.\n        If non-None, data with this value will be replaced with NaN so\n        that it is transparent (by default) when plotted.')
    padding = param.ClassSelector(default=0, class_=(int, float, tuple))
    show_legend = param.Boolean(default=False, doc='\n        Whether to show legend for the plot.')
    situate_axes = param.Boolean(default=True, doc='\n        Whether to situate the image relative to other plots. ')
    _plot_methods = dict(single='imshow')

    def get_extents(self, element, ranges, range_type='combined', **kwargs):
        extents = super().get_extents(element, ranges, range_type)
        if self.situate_axes or range_type not in ('combined', 'data'):
            return extents
        elif isinstance(element, Image):
            return element.bounds.lbrt()
        else:
            return element.extents

    def _compute_ticks(self, element, ranges):
        return (None, None)