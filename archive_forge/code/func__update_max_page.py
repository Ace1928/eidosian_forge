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
def _update_max_page(self):
    length = self._length
    nrows = self.page_size
    max_page = max(length // nrows + bool(length % nrows), 1)
    self.param.page.bounds = (1, max_page)
    for ref, (model, _) in self._models.items():
        self._apply_update([], {'max_page': max_page}, model, ref)