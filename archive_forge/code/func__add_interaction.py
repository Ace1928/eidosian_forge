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
def _add_interaction(int_type, **kwargs):
    """Add the interaction for the specified type.

    If a figure is passed using the key-word argument `figure` it is used. Else
    the context figure is used.
    If a list of marks are passed using the key-word argument `marks` it
    is used. Else the latest mark that is passed is used as the only mark
    associated with the selector.

    Parameters
    ----------
    int_type: type
        The type of interaction to be added.
    """
    fig = kwargs.pop('figure', current_figure())
    marks = kwargs.pop('marks', [_context['last_mark']])
    for name, traitlet in int_type.class_traits().items():
        dimension = traitlet.get_metadata('dimension')
        if dimension is not None:
            kwargs[name] = _get_context_scale(dimension)
    kwargs['marks'] = marks
    interaction = int_type(**kwargs)
    if fig.interaction is not None:
        fig.interaction.close()
    fig.interaction = interaction
    return interaction