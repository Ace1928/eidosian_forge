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
class DatetimeInput(LiteralInput):
    """
    The `DatetimeInput` allows specifying Python `datetime` like values using
    a text input widget.

    An optional `type` may be declared.

    Reference: https://panel.holoviz.org/reference/widgets/DatetimeInput.html

    :Example:

    >>> DatetimeInput(name='Datetime', value=datetime(2019, 2, 8))
    """
    value = param.Date(default=None, doc='\n        The current value')
    start = param.Date(default=None, doc='\n        Inclusive lower bound of the allowed date selection')
    end = param.Date(default=None, doc='\n        Inclusive upper bound of the allowed date selection')
    format = param.String(default='%Y-%m-%d %H:%M:%S', doc='\n        Datetime format used for parsing and formatting the datetime.')
    type = datetime
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'value': None, 'start': None, 'end': None}
    _rename: ClassVar[Mapping[str, str | None]] = {'format': None, 'type': None, 'start': None, 'end': None, 'serializer': None}

    def __init__(self, **params):
        super().__init__(**params)
        self.param.watch(self._validate, 'value')
        self._validate(None)

    def _validate(self, event):
        new = self.value
        if new is not None and (self.start is not None and self.start > new or (self.end is not None and self.end < new)):
            value = datetime.strftime(new, self.format)
            start = datetime.strftime(self.start, self.format)
            end = datetime.strftime(self.end, self.format)
            if event:
                self.value = event.old
            raise ValueError('DatetimeInput value must be between {start} and {end}, supplied value is {value}'.format(start=start, end=end, value=value))

    def _process_property_change(self, msg):
        msg = Widget._process_property_change(self, msg)
        new_state = ''
        if 'value' in msg:
            value = msg.pop('value')
            try:
                value = datetime.strptime(value, self.format)
            except Exception:
                new_state = ' (invalid)'
                value = self.value
            else:
                if value is not None and (self.start is not None and self.start > value or (self.end is not None and self.end < value)):
                    new_state = ' (out of bounds)'
                    value = self.value
            msg['value'] = value
        msg['name'] = msg.get('title', self.name).replace(self._state, '') + new_state
        self._state = new_state
        return msg

    def _process_param_change(self, msg):
        msg = Widget._process_param_change(self, msg)
        if 'value' in msg:
            value = msg['value']
            if value is None:
                value = ''
            else:
                value = datetime.strftime(msg['value'], self.format)
            msg['value'] = value
        msg['title'] = self.name
        return msg