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
def _serialize_value(self, value):
    if isinstance(value, str) and value:
        value = [np.datetime64(value) if self.as_numpy_datetime64 else datetime.strptime(value, '%Y-%m-%d %H:%M:%S') for value in value.split(' to ')]
        value = tuple(value)
    return value