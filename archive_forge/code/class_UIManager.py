import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class UIManager(Gtk.UIManager):

    def add_ui_from_string(self, buffer):
        if not isinstance(buffer, str):
            raise TypeError('buffer must be a string')
        length = _get_utf8_length(buffer)
        return Gtk.UIManager.add_ui_from_string(self, buffer, length)

    def insert_action_group(self, buffer, length=-1):
        return Gtk.UIManager.insert_action_group(self, buffer, length)