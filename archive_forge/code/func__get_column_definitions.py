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
def _get_column_definitions(self, col_names: List[str], df: pd.DataFrame) -> List[TableColumn]:
    import pandas as pd
    indexes = self.indexes
    columns = []
    for col in col_names:
        if col in df.columns:
            data = df[col]
        elif col in self.indexes:
            if len(self.indexes) == 1:
                data = df.index
            else:
                data = df.index.get_level_values(self.indexes.index(col))
        if isinstance(data, pd.DataFrame):
            raise ValueError('DataFrame contains duplicate column names.')
        col_kwargs = {}
        kind = data.dtype.kind
        editor: CellEditor
        formatter: CellFormatter | None = self.formatters.get(col)
        if kind == 'i':
            editor = IntEditor()
        elif kind == 'b':
            editor = CheckboxEditor()
        elif kind == 'f':
            editor = NumberEditor()
        elif isdatetime(data) or kind == 'M':
            editor = DateEditor()
        else:
            editor = StringEditor()
        if col in self.editors and (not isinstance(self.editors[col], (dict, str))):
            editor = self.editors[col]
            if isinstance(editor, CellEditor):
                editor = clone_model(editor)
        if col in indexes or editor is None:
            editor = CellEditor()
        if formatter is None or isinstance(formatter, (dict, str)):
            if kind == 'i':
                formatter = NumberFormatter(text_align='right')
            elif kind == 'b':
                formatter = StringFormatter(text_align='center')
            elif kind == 'f':
                formatter = NumberFormatter(format='0,0.0[00000]', text_align='right')
            elif isdatetime(data) or kind == 'M':
                if len(data) and isinstance(data.values[0], dt.date):
                    date_format = '%Y-%m-%d'
                else:
                    date_format = '%Y-%m-%d %H:%M:%S'
                formatter = DateFormatter(format=date_format, text_align='right')
            else:
                formatter = StringFormatter()
            default_text_align = True
        else:
            if isinstance(formatter, CellFormatter):
                formatter = clone_model(formatter)
            if hasattr(formatter, 'text_align'):
                default_text_align = type(formatter).text_align.class_default(formatter) == formatter.text_align
            else:
                default_text_align = True
        if not hasattr(formatter, 'text_align'):
            pass
        elif isinstance(self.text_align, str):
            formatter.text_align = self.text_align
            if not default_text_align:
                msg = f"The 'text_align' in Tabulator.formatters[{col!r}] is overridden by Tabulator.text_align"
                warn(msg, RuntimeWarning)
        elif col in self.text_align:
            formatter.text_align = self.text_align[col]
            if not default_text_align:
                msg = f"The 'text_align' in Tabulator.formatters[{col!r}] is overridden by Tabulator.text_align[{col!r}]"
                warn(msg, RuntimeWarning)
        elif col in self.indexes:
            formatter.text_align = 'left'
        if isinstance(self.widths, int):
            col_kwargs['width'] = self.widths
        elif str(col) in self.widths and isinstance(self.widths.get(str(col)), int):
            col_kwargs['width'] = self.widths.get(str(col))
        else:
            col_kwargs['width'] = 0
        title = self.titles.get(col, str(col))
        if col in indexes and len(indexes) > 1 and self.hierarchical:
            title = 'Index: %s' % ' | '.join(indexes)
        elif col in self.indexes and col.startswith('level_'):
            title = ''
        column = TableColumn(field=str(col), title=title, editor=editor, formatter=formatter, **col_kwargs)
        columns.append(column)
    return columns