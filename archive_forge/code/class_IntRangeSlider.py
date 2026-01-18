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
class IntRangeSlider(RangeSlider):
    """
    The IntRangeSlider widget allows selecting an integer range using
    a slider with two handles.

    Reference: https://panel.holoviz.org/reference/widgets/IntRangeSlider.html

    :Example:

    >>> IntRangeSlider(
    ...     value=(2, 4), start=0, end=10, step=2, name="A tuple of integers"
    ... )
    """
    start = param.Integer(default=0, doc='\n        The lower bound.')
    end = param.Integer(default=1, doc='\n        The upper bound.')
    step = param.Integer(default=1, doc='\n        The step size')

    def _process_property_change(self, msg):
        msg = super()._process_property_change(msg)
        if 'value' in msg:
            msg['value'] = tuple([v if v is None else round(v) for v in msg['value']])
        if 'value_throttled' in msg:
            msg['value_throttled'] = tuple([v if v is None else round(v) for v in msg['value_throttled']])
        return msg