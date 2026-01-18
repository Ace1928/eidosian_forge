import sys
from collections import OrderedDict
from IPython.display import display
from ipywidgets import VBox
from ipywidgets import Image as ipyImage
from numpy import arange, issubdtype, array, column_stack, shape
from .figure import Figure
from .scales import Scale, LinearScale, Mercator
from .axes import Axis
from .marks import (Lines, Scatter, ScatterGL, Hist, Bars, OHLC, Pie, Map, Image,
from .toolbar import Toolbar
from .interacts import (BrushIntervalSelector, FastIntervalSelector,
from traitlets.utils.sentinel import Sentinel
import functools
def _update_fig_axis_registry(fig, dimension, scale, axis):
    axis_registry = fig.axis_registry
    dimension_scales = axis_registry.get(dimension, [])
    dimension_scales.append({'scale': scale, 'axis': axis})
    axis_registry[dimension] = dimension_scales
    setattr(fig, 'axis_registry', axis_registry)