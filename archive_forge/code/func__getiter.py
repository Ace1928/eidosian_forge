import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def _getiter(self, key):
    if isinstance(key, Gtk.TreeIter):
        return key
    elif isinstance(key, int) and key < 0:
        index = len(self) + key
        if index < 0:
            raise IndexError('row index is out of bounds: %d' % key)
        return self.get_iter(index)
    else:
        try:
            aiter = self.get_iter(key)
        except ValueError:
            raise IndexError("could not find tree path '%s'" % key)
        return aiter