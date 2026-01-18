from __future__ import annotations
import asyncio
import math
import os
import sys
import time
from math import pi
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource, FixedTicker, Tooltip
from bokeh.plotting import figure
from tqdm.asyncio import tqdm as _tqdm
from .._param import Align
from ..io.resources import CDN_DIST
from ..layout import Column, Panel, Row
from ..models import (
from ..pane.markup import Str
from ..reactive import SyncableData
from ..util import PARAM_NAME_PATTERN, escape, updating
from ..viewable import Viewable
from .base import Widget
@property
def _color_intervals(self):
    vmin, vmax = self.bounds
    value = self.value
    ncolors = len(self.colors) if self.colors else 1
    interval = vmax - vmin
    if math.isfinite(value):
        fraction = value / interval
        idx = round(fraction * (ncolors - 1))
    else:
        fraction = 0
        idx = 0
    if not self.colors:
        intervals = [(fraction, self.default_color)]
        intervals.append((1, self.unfilled_color))
    elif self.show_boundaries:
        intervals = [c if isinstance(c, tuple) else ((i + 1) / ncolors, c) for i, c in enumerate(self.colors)]
    else:
        intervals = [self.colors[idx] if isinstance(self.colors[0], tuple) else (fraction, self.colors[idx])]
        intervals.append((1, self.unfilled_color))
    return intervals