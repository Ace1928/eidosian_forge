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
def _init_widget(self, i, options):
    """
        Helper method to initialize a select widget.
        """
    if isinstance(options, dict):
        options = list(options.keys())
    elif not isinstance(options, (list, dict)) and (not callable(options)):
        raise ValueError(f'options must be a dict, list, or callable that returns those types, got {options!r}, which is a {type(options).__name__}')
    widget_type, widget_kwargs = self._extract_level_metadata(i)
    value = self._lookup_value(i, options, self.value, error=False)
    widget_kwargs['options'] = options
    widget_kwargs['value'] = value
    if 'visible' not in widget_kwargs:
        widget_kwargs['visible'] = i == 0 or callable(options) or len(options) > 0
    widget = widget_type(**widget_kwargs)
    self.link(widget, disabled='disabled')
    widget.param.watch(self._update_widget_options_interactively, 'value')
    self._widgets.append(widget)
    return value