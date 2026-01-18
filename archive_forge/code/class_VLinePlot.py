import matplotlib as mpl
import numpy as np
import pandas as pd
import param
from matplotlib import patches
from matplotlib.lines import Line2D
from ...core.options import abbreviated_exception
from ...core.util import match_spec
from ...element import HLines, HSpans, VLines, VSpans
from .element import ColorbarPlot, ElementPlot
from .plot import mpl_rc_context
class VLinePlot(AnnotationPlot):
    """Draw a vertical line on the axis"""
    style_opts = ['alpha', 'color', 'linewidth', 'linestyle', 'visible']

    def draw_annotation(self, axis, position, opts):
        if self.invert_axes:
            return [axis.axhline(position, **opts)]
        else:
            return [axis.axvline(position, **opts)]