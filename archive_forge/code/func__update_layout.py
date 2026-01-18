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
@param.depends('object')
def _update_layout(self):
    if self.object is None:
        self._layout_panel = None
    else:
        self._layout_panel = self.object.layout()
        self._layout_panel.param.update(**{p: getattr(self, p) for p in Layoutable.param if p != 'name'})