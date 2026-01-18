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
def _validate_init_bounds(self, params):
    """
        This updates the default value, start and end
        if outside the fixed_start and fixed_end
        """
    start, end = (None, None)
    if 'start' not in params:
        if 'fixed_start' in params:
            start = params['fixed_start']
        elif 'end' in params:
            start = params.get('end') - params.get('step', 1)
        elif 'fixed_end' in params:
            start = params.get('fixed_end') - params.get('step', 1)
    if 'end' not in params:
        if 'fixed_end' in params:
            end = params['fixed_end']
        elif 'start' in params:
            end = params['start'] + params.get('step', 1)
        elif 'fixed_start' in params:
            end = params['fixed_start'] + params.get('step', 1)
    if start is not None:
        params['start'] = start
    if end is not None:
        params['end'] = end
    if 'value' not in params and 'start' in params:
        start = params['start']
        end = params.get('end', start + params.get('step', 1))
        params['value'] = (start, end)
    if 'value' not in params and 'end' in params:
        end = params['end']
        start = params.get('start', end - params.get('step', 1))
        params['value'] = (start, end)