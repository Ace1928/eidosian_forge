from a list of options.
from __future__ import annotations
import itertools
import re
from types import FunctionType
from typing import (
import numpy as np
import param
from bokeh.models import PaletteSelect
from bokeh.models.widgets import (
from ..io.resources import CDN_DIST
from ..layout.base import Column, ListPanel, NamedListPanel
from ..models import (
from ..util import PARAM_NAME_PATTERN, indexOf, isIn
from ._mixin import TooltipMixin
from .base import CompositeWidget, Widget
from .button import Button, _ButtonBase
from .input import TextAreaInput, TextInput
class _MultiSelectBase(SingleSelectBase):
    value = param.List(default=[])
    width = param.Integer(default=300, allow_None=True, doc='\n      Width of this component. If sizing_mode is set to stretch\n      or scale mode this will merely be used as a suggestion.')
    description = param.String(default=None, doc='\n        An HTML string describing the function of this component.')
    _supports_embed: ClassVar[bool] = False
    __abstract = True

    def _process_param_change(self, msg):
        msg = super(SingleSelectBase, self)._process_param_change(msg)
        labels, values = (self.labels, self.values)
        if 'value' in msg:
            msg['value'] = [labels[indexOf(v, values)] for v in msg['value'] if isIn(v, values)]
        if 'options' in msg:
            msg['options'] = labels
            if any((not isIn(v, values) for v in self.value)):
                self.value = [v for v in self.value if isIn(v, values)]
        return msg

    def _process_property_change(self, msg):
        msg = super(SingleSelectBase, self)._process_property_change(msg)
        if 'value' in msg:
            labels = self.labels
            msg['value'] = [self._items[v] for v in msg['value'] if v in labels]
        msg.pop('options', None)
        return msg