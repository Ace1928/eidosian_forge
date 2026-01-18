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
def _infer_x_for_line(y):
    """
    Infers the x for a line if no x is provided.
    """
    array_shape = shape(y)
    if len(array_shape) == 0:
        return []
    if len(array_shape) == 1:
        return arange(array_shape[0])
    if len(array_shape) > 1:
        return arange(array_shape[1])