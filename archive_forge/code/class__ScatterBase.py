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
class _ScatterBase(Mark):
    """
    Base Mark for Label and Scatter
    """
    x = Array([], allow_none=True).tag(sync=True, scaled=True, rtype='Number', atype='bqplot.Axis', **array_serialization).valid(array_dimension_bounds(1, 1))
    y = Array([], allow_none=True).tag(sync=True, scaled=True, rtype='Number', atype='bqplot.Axis', **array_serialization).valid(array_dimension_bounds(1, 1))
    color = Array(None, allow_none=True).tag(sync=True, scaled=True, rtype='Color', atype='bqplot.ColorAxis', **array_serialization).valid(array_squeeze, array_dimension_bounds(1, 1))
    opacity = Array(None, allow_none=True).tag(sync=True, scaled=True, rtype='Number', **array_serialization).valid(array_squeeze, array_dimension_bounds(1, 1))
    size = Array(None, allow_none=True).tag(sync=True, scaled=True, rtype='Number', **array_serialization).valid(array_squeeze, array_dimension_bounds(1, 1))
    rotation = Array(None, allow_none=True).tag(sync=True, scaled=True, rtype='Number', **array_serialization).valid(array_squeeze, array_dimension_bounds(1, 1))
    scales_metadata = Dict({'x': {'orientation': 'horizontal', 'dimension': 'x'}, 'y': {'orientation': 'vertical', 'dimension': 'y'}, 'color': {'dimension': 'color'}, 'size': {'dimension': 'size'}, 'opacity': {'dimension': 'opacity'}, 'rotation': {'dimension': 'rotation'}}).tag(sync=True)
    opacities = Array([1.0]).tag(sync=True, display_name='Opacities', **array_serialization).valid(array_squeeze, array_dimension_bounds(1, 1))
    hovered_style = Dict().tag(sync=True)
    unhovered_style = Dict().tag(sync=True)
    hovered_point = Int(None, allow_none=True).tag(sync=True)
    enable_move = Bool().tag(sync=True)
    enable_delete = Bool().tag(sync=True)
    restrict_x = Bool().tag(sync=True)
    restrict_y = Bool().tag(sync=True)
    update_on_move = Bool().tag(sync=True)

    def __init__(self, **kwargs):
        self._drag_start_handlers = CallbackDispatcher()
        self._drag_handlers = CallbackDispatcher()
        self._drag_end_handlers = CallbackDispatcher()
        super(_ScatterBase, self).__init__(**kwargs)
        self._name_to_handler.update({'drag_start': self._drag_start_handlers, 'drag_end': self._drag_end_handlers, 'drag': self._drag_handlers})

    def on_drag_start(self, callback, remove=False):
        self._drag_start_handlers.register_callback(callback, remove=remove)

    def on_drag(self, callback, remove=False):
        self._drag_handlers.register_callback(callback, remove=remove)

    def on_drag_end(self, callback, remove=False):
        self._drag_end_handlers.register_callback(callback, remove=remove)