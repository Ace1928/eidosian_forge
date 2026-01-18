import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class IconView(Gtk.IconView):
    if GTK3:
        __init__ = deprecated_init(Gtk.IconView.__init__, arg_names=('model',), category=PyGTKDeprecationWarning)
    get_item_at_pos = strip_boolean_result(Gtk.IconView.get_item_at_pos)
    get_visible_range = strip_boolean_result(Gtk.IconView.get_visible_range)
    get_dest_item_at_pos = strip_boolean_result(Gtk.IconView.get_dest_item_at_pos)