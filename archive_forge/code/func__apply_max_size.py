from __future__ import annotations
import datetime as dt
import uuid
from functools import partial
from types import FunctionType, MethodType
from typing import (
import numpy as np
import param
from bokeh.model import Model
from bokeh.models import ColumnDataSource, ImportedStyleSheet
from bokeh.models.widgets.tables import (
from bokeh.util.serialization import convert_datetime_array
from pyviz_comms import JupyterComm
from ..depends import transform_reference
from ..io.resources import CDN_DIST, CSS_URLS
from ..io.state import state
from ..reactive import Reactive, ReactiveData
from ..util import (
from ..util.warnings import warn
from .base import Widget
from .button import Button
from .input import TextInput
@param.depends('value', watch=True, on_init=True)
def _apply_max_size(self):
    """
        Ensure large tables automatically enable remote pagination.
        """
    if self.value is None or self._explicit_pagination:
        return
    with param.parameterized.discard_events(self):
        if self.hierarchical:
            pass
        elif self._MAX_ROW_LIMITS[0] < len(self.value) <= self._MAX_ROW_LIMITS[1]:
            self.pagination = 'local'
        elif len(self.value) > self._MAX_ROW_LIMITS[1]:
            self.pagination = 'remote'
    self._explicit_pagination = False