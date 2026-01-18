import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def _set_lists(cols, vals):
    if len(cols) != len(vals):
        raise TypeError('The number of columns do not match the number of values')
    columns = []
    values = []
    for col_num, value in zip(cols, vals):
        if not isinstance(col_num, int):
            raise TypeError('TypeError: Expected integer argument for column.')
        columns.append(col_num)
        values.append(self._convert_value(col_num, value))
    Gtk.TreeStore.set(self, treeiter, columns, values)