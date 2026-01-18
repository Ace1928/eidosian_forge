from __future__ import annotations
import ast
import json
from base64 import b64decode
from datetime import date, datetime
from typing import (
import numpy as np
import param
from bokeh.models.formatters import TickFormatter
from bokeh.models.widgets import (
from ..config import config
from ..layout import Column, Panel
from ..models import (
from ..util import param_reprs, try_datetime64_to_datetime
from .base import CompositeWidget, Widget
@param.depends('start', 'end', watch=True)
def _update_value_bounds(self):
    self.param.value.bounds = (self._convert_to_datetime(self.start), self._convert_to_datetime(self.end))
    self.param.value._validate(self._convert_to_datetime(self.value))