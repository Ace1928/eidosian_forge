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
class StaticText(Widget):
    """
    The `StaticText` widget displays a text value, but does not allow editing
    it.

    Reference: https://panel.holoviz.org/reference/widgets/StaticText.html

    :Example:

    >>> StaticText(name='Model', value='animagen2')
    """
    value = param.Parameter(default=None, doc='\n        The current value')
    _format: ClassVar[str] = '<b>{title}</b>: {value}'
    _rename: ClassVar[Mapping[str, str | None]] = {'name': None, 'value': 'text'}
    _target_transforms: ClassVar[Mapping[str, str | None]] = {'value': 'target.text.split(": ")[0]+": "+value'}
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'value': 'value.split(": ")[1]'}
    _widget_type: ClassVar[Type[Model]] = _BkDiv

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        if 'text' in msg:
            text = str(msg.pop('text'))
            partial = self._format.replace('{value}', '').format(title=self.name)
            if self.name:
                text = self._format.format(title=self.name, value=text.replace(partial, ''))
            msg['text'] = text
        return msg