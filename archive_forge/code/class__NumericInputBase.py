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
class _NumericInputBase(Widget):
    description = param.String(default=None, doc='\n        An HTML string describing the function of this component.')
    value = param.Number(default=0, allow_None=True, doc='\n        The current value of the spinner.')
    placeholder = param.String(default='0', doc='\n        Placeholder for empty input field.')
    format = param.ClassSelector(default=None, class_=(str, TickFormatter), doc='\n        Allows defining a custom format string or bokeh TickFormatter.')
    start = param.Parameter(default=None, allow_None=True, doc='\n        Optional minimum allowable value.')
    end = param.Parameter(default=None, allow_None=True, doc='\n        Optional maximum allowable value.')
    _rename: ClassVar[Mapping[str, str | None]] = {'start': 'low', 'end': 'high'}
    _widget_type: ClassVar[Type[Model]] = _BkNumericInput
    __abstract = True