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
def _get_pane(self, backend, state, **kwargs):
    pane_type = self._panes.get(backend, panel)
    if isinstance(pane_type, type):
        if issubclass(pane_type, Matplotlib):
            kwargs['tight'] = True
            kwargs['format'] = self.format
        if issubclass(pane_type, Bokeh):
            kwargs['autodispatch'] = False
    return pane_type(state, **kwargs)