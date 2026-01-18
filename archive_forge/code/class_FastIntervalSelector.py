from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
@register_interaction('bqplot.FastIntervalSelector')
class FastIntervalSelector(OneDSelector):
    """Fast interval selector interaction.

    This 1-D selector is used to select an interval on the x-scale
    by just moving the mouse (without clicking or dragging). The
    x-coordinate of the mouse controls the mid point of the interval selected
    while the y-coordinate of the mouse controls the the width of the interval.
    The larger the y-coordinate, the wider the interval selected.

    Interval selector has three modes:
        1. default mode: This is the default mode in which the mouse controls
                the location and width of the interval.
        2. fixed-width mode: In this mode the width of the interval is frozen
                and only the location of the interval is controlled with the
                mouse.
                A single click from the default mode takes you to this mode.
                Another single click takes you back to the default mode.
        3. frozen mode: In this mode the selected interval is frozen and the
                selector does not respond to mouse move.
                A double click from the default mode takes you to this mode.
                Another double click takes you back to the default mode.

    Attributes
    ----------
    selected: numpy.ndarray
        Two-element array containing the start and end of the interval selected
        in terms of the scale of the selector.
    color: Color or None (default: None)
        color of the rectangle representing the interval selector
    size: Float or None (default: None)
        if not None, this is the fixed pixel-width of the interval selector
    """
    selected = Array(None, allow_none=True).tag(sync=True, **array_serialization)
    color = Color(None, allow_none=True).tag(sync=True)
    size = Float(None, allow_none=True).tag(sync=True)
    _view_name = Unicode('FastIntervalSelector').tag(sync=True)
    _model_name = Unicode('FastIntervalSelectorModel').tag(sync=True)