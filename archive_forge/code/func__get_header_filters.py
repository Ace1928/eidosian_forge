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
def _get_header_filters(self, df):
    filters = []
    for filt in getattr(self, 'filters', []):
        col_name = filt['field']
        op = filt['type']
        val = filt['value']
        filt_def = getattr(self, 'header_filters', {}) or {}
        if col_name in df.columns:
            col = df[col_name]
        elif col_name in self.indexes:
            if len(self.indexes) == 1:
                col = df.index
            else:
                col = df.index.get_level_values(self.indexes.index(col_name))
        else:
            continue
        if isinstance(val, list):
            if len(val) == 1:
                val = val[0]
            elif not val:
                continue
        val = col.dtype.type(val)
        if op == '=':
            filters.append(col == val)
        elif op == '!=':
            filters.append(col != val)
        elif op == '<':
            filters.append(col < val)
        elif op == '>':
            filters.append(col > val)
        elif op == '>=':
            filters.append(col >= val)
        elif op == '<=':
            filters.append(col <= val)
        elif op == 'in':
            if not isinstance(val, (list, np.ndarray)):
                val = [val]
            filters.append(col.isin(val))
        elif op == 'like':
            filters.append(col.str.contains(val, case=False, regex=False))
        elif op == 'starts':
            filters.append(col.str.startsWith(val))
        elif op == 'ends':
            filters.append(col.str.endsWith(val))
        elif op == 'keywords':
            match_all = filt_def.get(col_name, {}).get('matchAll', False)
            sep = filt_def.get(col_name, {}).get('separator', ' ')
            matches = val.split(sep)
            if match_all:
                for match in matches:
                    filters.append(col.str.contains(match, case=False, regex=False))
            else:
                filt = col.str.contains(matches[0], case=False, regex=False)
                for match in matches[1:]:
                    filt |= col.str.contains(match, case=False, regex=False)
                filters.append(filt)
        elif op == 'regex':
            raise ValueError('Regex filtering not supported.')
        else:
            raise ValueError(f'Filter type {op!r} not recognized.')
    return filters