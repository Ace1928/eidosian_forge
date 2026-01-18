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
class ColorMap(SingleSelectBase):
    """
    The `ColorMap` widget allows selecting a value from a dictionary of
    `options` each containing a colormap specified as a list of colors
    or a matplotlib colormap.

    Reference: https://panel.holoviz.org/reference/widgets/ColorMap.html

    :Example:

    >>> ColorMap(name='Reds', options={'Reds': ['white', 'red'], 'Blues': ['#ffffff', '#0000ff']})
    """
    options = param.Dict(default={}, doc='\n        Dictionary of colormaps')
    ncols = param.Integer(default=1, doc='\n        Number of columns of swatches to display.')
    swatch_height = param.Integer(default=20, doc='\n        Height of the color swatches.')
    swatch_width = param.Integer(default=100, doc='\n        Width of the color swatches.')
    value = param.Parameter(default=None, doc='The selected colormap.')
    value_name = param.String(default=None, doc='Name of the selected colormap.')
    _rename = {'options': 'items', 'value_name': None}
    _widget_type: ClassVar[Type[Model]] = PaletteSelect

    @param.depends('value_name', watch=True, on_init=True)
    def _sync_value_name(self):
        if self.value_name and self.value_name in self.options:
            self.value = self.options[self.value_name]

    @param.depends('value', watch=True, on_init=True)
    def _sync_value(self):
        if self.value:
            idx = indexOf(self.value, self.values)
            self.value_name = self.labels[idx]

    def _process_param_change(self, params):
        if 'options' in params:
            options = []
            for name, cmap in params.pop('options').items():
                if 'matplotlib' in getattr(cmap, '__module__', ''):
                    N = getattr(cmap, 'N', 10)
                    samples = np.linspace(0, 1, N)
                    rgba_tmpl = 'rgba({0}, {1}, {2}, {3:.3g})'
                    cmap = [rgba_tmpl.format(*(rgba[:3] * 255).astype(int), rgba[-1]) for rgba in cmap(samples)]
                options.append((name, cmap))
            params['options'] = options
        if 'value' in params and (not isinstance(params['value'], (str, type(None)))):
            idx = indexOf(params['value'], self.values)
            params['value'] = self.labels[idx]
        return {self._property_mapping.get(p, p): v for p, v in params.items() if self._property_mapping.get(p, False) is not None}