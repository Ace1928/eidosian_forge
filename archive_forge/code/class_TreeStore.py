import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class TreeStore(Gtk.TreeStore, TreeModel, TreeSortable):

    def __init__(self, *column_types):
        Gtk.TreeStore.__init__(self)
        self.set_column_types(column_types)

    def _do_insert(self, parent, position, row):
        if row is not None:
            row, columns = self._convert_row(row)
            treeiter = self.insert_with_values(parent, position, columns, row)
        else:
            treeiter = Gtk.TreeStore.insert(self, parent, position)
        return treeiter

    def append(self, parent, row=None):
        return self._do_insert(parent, -1, row)

    def prepend(self, parent, row=None):
        return self._do_insert(parent, 0, row)

    def insert(self, parent, position, row=None):
        return self._do_insert(parent, position, row)

    def insert_before(self, parent, sibling, row=None):
        if row is not None:
            if sibling is None:
                position = -1
            else:
                if parent is None:
                    parent = self.iter_parent(sibling)
                position = self.get_path(sibling).get_indices()[-1]
            return self._do_insert(parent, position, row)
        return Gtk.TreeStore.insert_before(self, parent, sibling)

    def insert_after(self, parent, sibling, row=None):
        if row is not None:
            if sibling is None:
                position = 0
            else:
                if parent is None:
                    parent = self.iter_parent(sibling)
                position = self.get_path(sibling).get_indices()[-1] + 1
            return self._do_insert(parent, position, row)
        return Gtk.TreeStore.insert_after(self, parent, sibling)

    def set_value(self, treeiter, column, value):
        value = self._convert_value(column, value)
        Gtk.TreeStore.set_value(self, treeiter, column, value)

    def set(self, treeiter, *args):

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
        if args:
            if isinstance(args[0], int):
                _set_lists(args[::2], args[1::2])
            elif isinstance(args[0], (tuple, list)):
                if len(args) != 2:
                    raise TypeError('Too many arguments')
                _set_lists(args[0], args[1])
            elif isinstance(args[0], dict):
                _set_lists(args[0].keys(), args[0].values())
            else:
                raise TypeError('Argument list must be in the form of (column, value, ...), ((columns,...), (values, ...)) or {column: value}.  No -1 termination is needed.')