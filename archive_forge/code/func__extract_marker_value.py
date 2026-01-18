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
def _extract_marker_value(marker_str, code_dict):
    """Extracts the marker value from a given marker string.

        Looks up the `code_dict` and returns the corresponding marker for a
        specific code.

        For example if `marker_str` is 'g-o' then the method extracts
        - 'green' if the code_dict is color_codes,
        - 'circle' if the code_dict is marker_codes etc.
        """
    val = None
    for code in code_dict:
        if code in marker_str:
            val = code_dict[code]
            break
    return val