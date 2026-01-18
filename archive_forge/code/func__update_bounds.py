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
@param.depends('start', 'end', 'fixed_start', 'fixed_end', watch=True)
def _update_bounds(self):
    self.param.value.softbounds = (self.start, self.end)
    self.param.value_throttled.softbounds = (self.start, self.end)
    self.param.value.bounds = (self.fixed_start, self.fixed_end)
    self.param.value_throttled.bounds = (self.fixed_start, self.fixed_end)
    if self.fixed_start is not None:
        self._slider.start = max(self.fixed_start, self.start)
    if self.fixed_end is not None:
        self._slider.end = min(self.fixed_end, self.end)
    self._start_edit.start = self.fixed_start
    self._start_edit.end = self.fixed_end
    self._end_edit.start = self.fixed_start
    self._end_edit.end = self.fixed_end