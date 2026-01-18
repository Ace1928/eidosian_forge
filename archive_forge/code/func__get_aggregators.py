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
def _get_aggregators(self, group):
    numeric_cols = list(self.value.select_dtypes(include='number').columns)
    aggs = self.aggregators.get(group, [])
    if not isinstance(aggs, list):
        aggs = [aggs]
    expanded_aggs = []
    for col_aggs in aggs:
        if not isinstance(col_aggs, dict):
            col_aggs = {col: col_aggs for col in numeric_cols}
        for col, agg in col_aggs.items():
            if isinstance(agg, str):
                agg = self._aggregators.get(agg)
            if issubclass(agg, RowAggregator):
                expanded_aggs.append(agg(field_=str(col)))
    return expanded_aggs