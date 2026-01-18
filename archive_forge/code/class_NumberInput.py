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
class NumberInput(_SpinnerBase):

    def __new__(self, **params):
        param_list = ['value', 'start', 'stop', 'step']
        if all((isinstance(params.get(p, 0), int) for p in param_list)):
            return IntInput(**params)
        else:
            return FloatInput(**params)