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
def _get_attribute_dimension(trait_name, mark_type=None):
    """Returns the dimension for the name of the trait for the specified mark.

    If `mark_type` is `None`, then the `trait_name` is returned
    as is.
    Returns `None` if the `trait_name` is not valid for `mark_type`.
    """
    if mark_type is None:
        return trait_name
    scale_metadata = mark_type.class_traits()['scales_metadata'].default_args[0]
    return scale_metadata.get(trait_name, {}).get('dimension', None)