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
def _track_overrides(self, *events):
    if self._syncing_props:
        return
    overrides = list(self._overrides)
    for e in events:
        if e.name in overrides and self.param[e.name].default == e.new:
            overrides.remove(e.name)
        else:
            overrides.append(e.name)
    self._overrides = overrides