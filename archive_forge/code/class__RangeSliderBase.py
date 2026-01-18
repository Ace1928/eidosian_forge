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
class _RangeSliderBase(_SliderBase):
    value = param.Tuple(length=2, allow_None=False, nested_refs=True, doc='\n        The selected range of the slider. Updated when a handle is dragged.')
    value_start = param.Parameter(readonly=True, doc='The lower value of the selected range.')
    value_end = param.Parameter(readonly=True, doc='The upper value of the selected range.')
    __abstract = True

    def __init__(self, **params):
        if 'value' not in params:
            params['value'] = (params.get('start', self.start), params.get('end', self.end))
        if params['value'] is not None:
            v1, v2 = params['value']
            params['value_start'], params['value_end'] = (resolve_value(v1), resolve_value(v2))
        with edit_readonly(self):
            super().__init__(**params)

    @param.depends('value', watch=True)
    def _sync_values(self):
        vs, ve = self.value
        with edit_readonly(self):
            self.param.update(value_start=vs, value_end=ve)

    def _process_property_change(self, msg):
        msg = super()._process_property_change(msg)
        if 'value' in msg:
            msg['value'] = tuple(msg['value'])
        if 'value_throttled' in msg:
            msg['value_throttled'] = tuple(msg['value_throttled'])
        return msg