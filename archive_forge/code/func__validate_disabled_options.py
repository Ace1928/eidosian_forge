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
def _validate_disabled_options(self, *events):
    if self.disabled_options and self.disabled_options == self.values:
        raise ValueError(f'All the options of a {type(self).__name__} widget cannot be disabled.')
    not_in_opts = [dopts for dopts in self.disabled_options if dopts not in (self.values or [])]
    if not_in_opts:
        raise ValueError(f'Cannot disable non existing options of {type(self).__name__}: {not_in_opts}')
    if len(events) == 1:
        if events[0].name == 'value' and self.value in self.disabled_options:
            raise ValueError(f'Cannot set the value of {type(self).__name__} to {self.value!r} as it is a disabled option.')
        elif events[0].name == 'disabled_options' and self.value in self.disabled_options:
            raise ValueError(f'Cannot set disabled_options of {type(self).__name__} to a list that includes the current value {self.value!r}.')
    if self.value in self.disabled_options:
        raise ValueError(f'Cannot initialize {type(self).__name__} with value {self.value!r} as it is one of the disabled options.')