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
def _sort_df(self, df: pd.DataFrame) -> pd.DataFrame:
    if not self.sorters:
        return df
    fields = [self._renamed_cols.get(s['field'], s['field']) for s in self.sorters]
    ascending = [s['dir'] == 'asc' for s in self.sorters]
    df['_index_'] = np.arange(len(df)).astype(str)
    fields.append('_index_')
    ascending.append(True)
    if self.show_index:
        rename = 'index' in fields and df.index.name is None
        if rename:
            df.index.name = 'index'
    else:
        rename = False

    def tabulator_sorter(col):
        if col.dtype.kind not in 'SUO':
            return col
        try:
            return col.fillna('').str.lower()
        except Exception:
            return col
    df_sorted = df.sort_values(fields, ascending=ascending, kind='mergesort', key=tabulator_sorter)
    if rename:
        df.index.name = None
        df_sorted.index.name = None
    df.drop(columns=['_index_'], inplace=True)
    df_sorted.drop(columns=['_index_'], inplace=True)
    return df_sorted