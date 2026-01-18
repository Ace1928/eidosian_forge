from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
@register_interaction('bqplot.HandDraw')
class HandDraw(Interaction):
    """A hand-draw interaction.

    This can be used to edit the 'y' value of an existing line using the mouse.
    The minimum and maximum x values of the line which can be edited may be
    passed as parameters.
    The y-values for any part of the line can be edited by drawing the desired
    path while holding the mouse-down.
    y-values corresponding to x-values smaller than min_x or greater than max_x
    cannot be edited by HandDraw.

    Attributes
    ----------
    lines: an instance Lines mark or None (default: None)
        The instance of Lines which is edited using the hand-draw interaction.
        The 'y' values of the line are changed according to the path of the
        mouse. If the lines has multi dimensional 'y', then the 'line_index'
        attribute is used to selected the 'y' to be edited.
    line_index: nonnegative integer (default: 0)
        For a line with multi-dimensional 'y', this indicates the index of the
        'y' to be edited by the handdraw.
    min_x: float or Date or None (default: None)
        The minimum value of 'x' which should be edited via the handdraw.
    max_x: float or Date or None (default: None)
        The maximum value of 'x' which should be edited via the handdraw.
    """
    lines = Instance(Lines, allow_none=True, default_value=None).tag(sync=True, **widget_serialization)
    line_index = Int().tag(sync=True)
    min_x = (Float(None, allow_none=True) | Date(None, allow_none=True)).tag(sync=True)
    max_x = (Float(None, allow_none=True) | Date(None, allow_none=True)).tag(sync=True)
    _view_name = Unicode('HandDraw').tag(sync=True)
    _model_name = Unicode('HandDrawModel').tag(sync=True)