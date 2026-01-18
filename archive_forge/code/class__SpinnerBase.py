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
class _SpinnerBase(_NumericInputBase):
    page_step_multiplier = param.Integer(default=10, bounds=(0, None), doc='\n        Defines the multiplication factor applied to step when the page up\n        and page down keys are pressed.')
    wheel_wait = param.Integer(default=100, doc='\n        Defines the debounce time in ms before updating `value_throttled` when\n        the mouse wheel is used to change the input.')
    width = param.Integer(default=300, allow_None=True, doc='\n      Width of this component. If sizing_mode is set to stretch\n      or scale mode this will merely be used as a suggestion.')
    _rename: ClassVar[Mapping[str, str | None]] = {'value_throttled': None}
    _widget_type: ClassVar[Type[Model]] = _BkSpinner
    __abstract = True

    def __init__(self, **params):
        if 'value' not in params:
            value = params.get('start', self.value)
            if value is not None:
                params['value'] = value
        if 'value' in params and 'value_throttled' in self.param:
            params['value_throttled'] = params['value']
        super().__init__(**params)

    def __repr__(self, depth=0):
        return '{cls}({params})'.format(cls=type(self).__name__, params=', '.join(param_reprs(self, ['value_throttled'])))

    @property
    def _linked_properties(self) -> Tuple[str]:
        return super()._linked_properties + ('value_throttled',)

    def _update_model(self, events: Dict[str, param.parameterized.Event], msg: Dict[str, Any], root: Model, model: Model, doc: Document, comm: Optional[Comm]) -> None:
        if 'value_throttled' in msg:
            del msg['value_throttled']
        return super()._update_model(events, msg, root, model, doc, comm)

    def _process_param_change(self, msg):
        if 'value' in msg and msg['value'] == float('-inf'):
            msg['value'] = None
            msg['value_throttled'] = None
        return super()._process_param_change(msg)

    def _process_property_change(self, msg):
        if config.throttled:
            if 'value' in msg:
                del msg['value']
            if 'value_throttled' in msg:
                msg['value'] = msg['value_throttled']
        return super()._process_property_change(msg)