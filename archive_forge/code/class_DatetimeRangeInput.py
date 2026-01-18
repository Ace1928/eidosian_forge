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
class DatetimeRangeInput(CompositeWidget):
    """
    The `DatetimeRangeInput` widget allows selecting a `datetime` range using
    two `DatetimeInput` widgets, which return a `tuple` range.

    Reference: https://panel.holoviz.org/reference/widgets/DatetimeRangeInput.html

    :Example:

    >>> DatetimeRangeInput(
    ...     name='Datetime Range',
    ...     value=(datetime(2017, 1, 1), datetime(2018, 1, 10)),
    ...     start=datetime(2017, 1, 1), end=datetime(2019, 1, 1),
    ... )
    """
    value = param.Tuple(default=(None, None), length=2, doc='\n        The current value')
    start = param.Date(default=None, doc='\n        Inclusive lower bound of the allowed date selection')
    end = param.Date(default=None, doc='\n        Inclusive upper bound of the allowed date selection')
    format = param.String(default='%Y-%m-%d %H:%M:%S', doc='\n        Datetime format used for parsing and formatting the datetime.')
    _composite_type: ClassVar[Type[Panel]] = Column

    def __init__(self, **params):
        self._text = StaticText(margin=(5, 0, 0, 0), styles={'white-space': 'nowrap'})
        self._start = DatetimeInput(sizing_mode='stretch_width', margin=(5, 0, 0, 0))
        self._end = DatetimeInput(sizing_mode='stretch_width', margin=(5, 0, 0, 0))
        if 'value' not in params:
            params['value'] = (params['start'], params['end'])
        super().__init__(**params)
        self._msg = ''
        self._composite.extend([self._text, self._start, self._end])
        self._updating = False
        self.param.watch(self._update_widgets, [p for p in self.param if p != 'name'])
        self._update_widgets()
        self._update_label()

    @param.depends('name', '_start.name', '_end.name', watch=True)
    def _update_label(self):
        self._text.value = f'{self.name}{self._start.name}{self._end.name}{self._msg}'

    @param.depends('_start.value', '_end.value', watch=True)
    def _update(self):
        if self._updating:
            return
        if self._start.value is not None and self._end.value is not None and (self._start.value > self._end.value):
            self._msg = ' (start of range must be <= end)'
            self._update_label()
            return
        elif self._msg:
            self._msg = ''
            self._update_label()
        try:
            self._updating = True
            self.value = (self._start.value, self._end.value)
        finally:
            self._updating = False

    def _update_widgets(self, *events):
        filters = [event.name for event in events] if events else list(self.param)
        if 'name' in filters:
            filters.remove('name')
        if self._updating:
            return
        try:
            self._updating = True
            params = {k: v for k, v in self.param.values().items() if k in filters}
            start_params = dict(params, value=self.value[0])
            end_params = dict(params, value=self.value[1])
            self._start.param.update(**start_params)
            self._end.param.update(**end_params)
        finally:
            self._updating = False