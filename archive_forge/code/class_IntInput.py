from __future__ import annotations
import ast
import json
from base64 import b64decode
from datetime import date, datetime
from typing import (
import numpy as np
import param
from bokeh.models.formatters import TickFormatter
from bokeh.models.widgets import (
from ..config import config
from ..layout import Column, Panel
from ..models import (
from ..util import param_reprs, try_datetime64_to_datetime
from .base import CompositeWidget, Widget
class IntInput(_SpinnerBase, _IntInputBase):
    """
    The `IntInput` allows selecting an integer value using a spinbox.

    It behaves like a slider except that lower and upper bounds are optional
    and a specific value can be entered. The value can be changed using the
    keyboard (up, down, page up, page down), mouse wheel and arrow buttons.

    Reference: https://panel.holoviz.org/reference/widgets/IntInput.html

    :Example:

    >>> IntInput(name='Value', value=100, start=0, end=1000, step=10)
    """
    step = param.Integer(default=1, doc='\n        The step size.')
    value_throttled = param.Integer(default=None, constant=True, doc='\n        The current value. Updates only on `<enter>` or when the widget looses focus.')
    _rename: ClassVar[Mapping[str, str | None]] = {'start': 'low', 'end': 'high'}