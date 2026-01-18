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
def _apply_selection(self, event):
    """
        Applies the current selection depending on which button was
        pressed.
        """
    selected = event.obj is self._buttons[True]
    new = {k: self._items[k] for k in self._selections[not selected]}
    old = self._lists[selected].options
    other = self._lists[not selected].options
    merged = {k: k for k in list(old) + list(new)}
    leftovers = {k: k for k in other if k not in new}
    self._lists[selected].options = merged if merged else {}
    self._lists[not selected].options = leftovers if leftovers else {}
    if len(self._lists[True].options):
        self._selected[-1] = self._lists[True]
    else:
        self._selected[-1] = self._placeholder
    self.value = [self._items[o] for o in self._lists[True].options if o != '']
    self._apply_filters()