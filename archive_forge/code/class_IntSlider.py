from __future__ import annotations
import datetime as dt
from typing import (
import numpy as np
import param
from bokeh.models import CustomJS
from bokeh.models.formatters import TickFormatter
from bokeh.models.widgets import (
from bokeh.models.widgets.sliders import NumericalSlider as _BkNumericalSlider
from param.parameterized import resolve_value
from ..config import config
from ..io import state
from ..io.resources import CDN_DIST
from ..layout import Column, Panel, Row
from ..util import (
from ..viewable import Layoutable
from ..widgets import FloatInput, IntInput
from .base import CompositeWidget, Widget
from .input import StaticText
class IntSlider(ContinuousSlider):
    """
    The IntSlider widget allows selecting an integer value within a
    set of bounds using a slider.

    Reference: https://panel.holoviz.org/reference/widgets/IntSlider.html

    :Example:

    >>> IntSlider(value=5, start=0, end=10, step=1, name="Integer Value")
    """
    start = param.Integer(default=0, doc='\n        The lower bound.')
    end = param.Integer(default=1, doc='\n        The upper bound.')
    step = param.Integer(default=1, doc='\n        The step size.')
    value = param.Integer(default=0, allow_None=True, doc='\n        The selected integer value of the slider. Updated when the handle is dragged.')
    value_throttled = param.Integer(default=None, constant=True, doc='\n        The value of the slider. Updated when the handle is released')
    _rename: ClassVar[Mapping[str, str | None]] = {'name': 'title', 'value_throttled': None}

    def _process_property_change(self, msg):
        msg = super()._process_property_change(msg)
        if 'value' in msg:
            msg['value'] = msg['value'] if msg['value'] is None else int(msg['value'])
        if 'value_throttled' in msg:
            throttled = msg['value_throttled']
            msg['value_throttled'] = throttled if throttled is None else int(throttled)
        return msg