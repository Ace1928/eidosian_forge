import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def _convert_negative_index(self, index):
    new_index = self.model.get_n_columns() + index
    if new_index < 0:
        raise IndexError('column index is out of bounds: %d' % index)
    return new_index