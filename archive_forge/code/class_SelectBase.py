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
class SelectBase(Widget):
    options = param.ClassSelector(default=[], class_=(dict, list))
    __abstract = True

    @property
    def labels(self):
        labels = []
        for o in self.options:
            if isinstance(o, param.Parameterized) and (not PARAM_NAME_PATTERN.match(o.name)):
                labels.append(o.name)
            else:
                labels.append(str(o))
        return labels

    @property
    def values(self):
        if isinstance(self.options, dict):
            return list(self.options.values())
        else:
            return self.options

    @property
    def _items(self):
        return dict(zip(self.labels, self.values))