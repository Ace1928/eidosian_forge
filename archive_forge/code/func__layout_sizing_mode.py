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
@property
def _layout_sizing_mode(self):
    if self._width_responsive and self._height_responsive:
        smode = 'stretch_both'
    elif self._width_responsive:
        smode = 'stretch_width'
    elif self._height_responsive:
        smode = 'stretch_height'
    else:
        smode = None
    return smode