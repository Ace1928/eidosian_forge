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
class ptqdm(_tqdm):

    def __init__(self, *args, **kwargs):
        self._indicator = kwargs.pop('indicator')
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None, bar_style=None):
        super().display(msg, pos)
        styles = self._indicator.text_pane.styles or {}
        if 'color' not in styles:
            color = self.colour or 'black'
            self._indicator.text_pane.styles = dict(styles, color=color)
        if self.total is not None and self.n is not None:
            self._indicator.max = int(self.total)
            self._indicator.value = int(self.n)
            self._indicator.text = self._to_text(**self.format_dict)
        return True

    def _to_text(self, n, total, **kwargs):
        return self.format_meter(n, total, **{**kwargs, 'ncols': 0})

    def close(self):
        super().close()
        if not self.leave:
            self._indicator.reset()
        return _tqdm