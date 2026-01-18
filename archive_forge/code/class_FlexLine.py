import os
import json
from warnings import warn
import ipywidgets as widgets
from ipywidgets import (Widget, DOMWidget, CallbackDispatcher,
from traitlets import (Int, Unicode, List, Enum, Dict, Bool, Float,
from traittypes import Array
from numpy import histogram
import numpy as np
from .scales import Scale, OrdinalScale, LinearScale
from .traits import (Date, array_serialization,
from ._version import __frontend_version__
from .colorschemes import CATEGORY10
@register_mark('bqplot.FlexLine')
class FlexLine(Mark):
    """Flexible Lines mark.

    In the case of the FlexLines mark, scales for 'x' and 'y' MUST be provided.
    Scales for the color and width data attributes are optional. In the case
    where another data attribute than 'x' or 'y' is provided but the
    corresponding scale is missing, the data attribute is ignored.

    Attributes
    ----------
    name: string (class-level attributes)
        user-friendly name of the mark
    colors: list of colors (default: CATEGORY10)
        List of colors for the Lines
    stroke_width: float (default: 1.5)
        Default stroke width of the Lines

    Data Attributes

    x: numpy.ndarray (default: [])
        abscissas of the data points (1d array)
    y: numpy.ndarray (default: [])
        ordinates of the data points (1d array)
    color: numpy.ndarray or None (default: None)
        Array controlling the color of the data points
    width: numpy.ndarray or None (default: None)
        Array controlling the widths of the Lines.
    """
    icon = 'fa-line-chart'
    name = 'Flexible lines'
    x = Array([]).tag(sync=True, scaled=True, rtype='Number', atype='bqplot.Axis', **array_serialization).valid(array_squeeze, array_dimension_bounds(1, 1))
    y = Array([]).tag(sync=True, scaled=True, rtype='Number', atype='bqplot.Axis', **array_serialization).valid(array_squeeze, array_dimension_bounds(1, 1))
    color = Array(None, allow_none=True).tag(sync=True, scaled=True, rtype='Color', atype='bqplot.ColorAxis', **array_serialization).valid(array_squeeze)
    width = Array(None, allow_none=True).tag(sync=True, scaled=True, rtype='Number', **array_serialization).valid(array_squeeze)
    scales_metadata = Dict({'x': {'orientation': 'horizontal', 'dimension': 'x'}, 'y': {'orientation': 'vertical', 'dimension': 'y'}, 'color': {'dimension': 'color'}}).tag(sync=True)
    stroke_width = Float(1.5).tag(sync=True, display_name='Stroke width')
    colors = List(trait=Color(default_value=None, allow_none=True), default_value=CATEGORY10).tag(sync=True)
    _view_name = Unicode('FlexLine').tag(sync=True)
    _model_name = Unicode('FlexLineModel').tag(sync=True)