import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def _convert_row(self, row):
    if isinstance(row, str):
        raise TypeError('Expected a list or tuple, but got str')
    n_columns = self.get_n_columns()
    if len(row) != n_columns:
        raise ValueError('row sequence has the incorrect number of elements')
    result = []
    columns = []
    for cur_col, value in enumerate(row):
        if value is None:
            continue
        result.append(self._convert_value(cur_col, value))
        columns.append(cur_col)
    return (result, columns)