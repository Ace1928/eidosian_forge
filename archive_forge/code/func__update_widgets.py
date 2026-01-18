from __future__ import annotations
import sys
from collections import defaultdict
from functools import partial
from typing import (
import param
from bokeh.models import Range1d, Spacer as _BkSpacer
from bokeh.themes.theme import Theme
from packaging.version import Version
from param.parameterized import register_reference_transform
from param.reactive import bind
from ..io import state, unlocked
from ..layout import (
from ..viewable import Layoutable, Viewable
from ..widgets import Player
from .base import PaneBase, RerenderError, panel
from .plot import Bokeh, Matplotlib
from .plotly import Plotly
@param.depends('widget_type', 'widgets', watch=True)
def _update_widgets(self, *events):
    if self.object is None:
        widgets, values = ([], [])
    else:
        direction = getattr(self.widget_layout, '_direction', 'vertical')
        widgets, values = self.widgets_from_dimensions(self.object, self.widgets, self.widget_type, direction)
    self._values = values
    for cb in list(self._internal_callbacks):
        if cb.inst in self.widget_box.objects:
            cb.inst.param.unwatch(cb)
            self._internal_callbacks.remove(cb)
    for widget in widgets:
        watcher = widget.param.watch(self._widget_callback, 'value')
        self._internal_callbacks.append(watcher)
    self.widget_box[:] = widgets
    if widgets and self.widget_box not in self._widget_container or (not widgets and self.widget_box in self._widget_container) or (not self._initialized):
        self._update_layout()