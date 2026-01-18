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
def _lookup_value(self, i, options, values, name=None, error=False):
    """
        Look up the value of the select widget at index i or by name.
        """
    options_iterable = isinstance(options, (list, dict))
    if values is None or (options_iterable and len(options) == 0):
        value = None
    elif name is None:
        value = list(values.values())[i] if i < len(values) else None
    elif isinstance(self._levels[0], int):
        value = values.get(i)
    else:
        value = values.get(name)
    if options_iterable and options and (value not in options):
        if value is not None and error:
            raise ValueError(f'Failed to set value {value!r} for level {name!r}, must be one of {options!r}.')
        else:
            value = options[0]
    return value