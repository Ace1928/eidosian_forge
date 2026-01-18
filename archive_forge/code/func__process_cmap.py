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
def _process_cmap(cmap):
    """
    Returns a kwarg dict suitable for a ColorScale
    """
    option = {}
    if isinstance(cmap, str):
        option['scheme'] = cmap
    elif isinstance(cmap, list):
        option['colors'] = cmap
    else:
        raise ValueError('`cmap` must be a string (name of a color scheme)\n                         or a list of colors, but a value of {} was given\n                         '.format(cmap))
    return option