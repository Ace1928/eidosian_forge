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
class ArrayInput(LiteralInput):
    """
    The `ArrayInput` allows rendering and editing NumPy arrays in a text
    input widget.

    Arrays larger than the `max_array_size` will be summarized and editing
    will be disabled.

    Reference: https://panel.holoviz.org/reference/widgets/ArrayInput.html

    :Example:

    >>> To be determined ...
    """
    max_array_size = param.Number(default=1000, doc='\n        Arrays larger than this limit will be allowed in Python but\n        will not be serialized into JavaScript. Although such large\n        arrays will thus not be editable in the widget, such a\n        restriction helps avoid overwhelming the browser and lets\n        other widgets remain usable.')
    _rename: ClassVar[Mapping[str, str | None]] = {'max_array_size': None}
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'serializer': None, 'type': None, 'value': None}

    def __init__(self, **params):
        super().__init__(**params)
        self._auto_disabled = False

    def _process_property_change(self, msg):
        msg = super()._process_property_change(msg)
        if 'value' in msg and isinstance(msg['value'], list):
            msg['value'] = np.asarray(msg['value'])
        return msg

    def _process_param_change(self, msg):
        if msg.get('disabled', False):
            self._auto_disabled = False
        value = msg.get('value')
        if value is None:
            return super()._process_param_change(msg)
        if value.size <= self.max_array_size:
            msg['value'] = value.tolist()
            if self.disabled and self._auto_disabled:
                self.disabled = False
                msg['disabled'] = False
                self._auto_disabled = False
        else:
            msg['value'] = np.array2string(msg['value'], separator=',', threshold=self.max_array_size)
            if not self.disabled:
                self.param.warning(f'Number of array elements ({value.size}) exceeds `max_array_size` ({self.max_array_size}), editing will be disabled.')
                self.disabled = True
                msg['disabled'] = True
                self._auto_disabled = True
        return super()._process_param_change(msg)