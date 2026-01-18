import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class TreeViewColumn(Gtk.TreeViewColumn):

    def __init__(self, title='', cell_renderer=None, **attributes):
        Gtk.TreeViewColumn.__init__(self, title=title)
        if cell_renderer:
            self.pack_start(cell_renderer, True)
        for name, value in attributes.items():
            self.add_attribute(cell_renderer, name, value)
    cell_get_position = strip_boolean_result(Gtk.TreeViewColumn.cell_get_position)

    def set_cell_data_func(self, cell_renderer, func, func_data=None):
        super(TreeViewColumn, self).set_cell_data_func(cell_renderer, func, func_data)

    def set_attributes(self, cell_renderer, **attributes):
        Gtk.CellLayout.clear_attributes(self, cell_renderer)
        for name, value in attributes.items():
            Gtk.CellLayout.add_attribute(self, cell_renderer, name, value)