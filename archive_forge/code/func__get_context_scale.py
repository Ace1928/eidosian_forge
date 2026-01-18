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
def _get_context_scale(dimension):
    """Return the scale instance in the current context for a given dimension.

    Parameters
    ----------

    dimension: string
        The dimension along which the current context scale is to be fetched.
    """
    return _context['scales'][dimension]