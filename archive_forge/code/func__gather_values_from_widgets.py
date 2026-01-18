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
def _gather_values_from_widgets(self, up_to_i=None):
    """
        Gather values from all the select widgets to update the class' value.
        """
    values = {}
    for i, select in enumerate(self._widgets):
        if up_to_i is not None and i >= up_to_i:
            break
        level = self._levels[i]
        if isinstance(level, dict):
            name = level.get('name', i)
        else:
            name = level
        values[name] = select.value if select.options else None
    return values