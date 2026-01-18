import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class TreeModelRowIter(object):

    def __init__(self, model, aiter):
        self.model = model
        self.iter = aiter

    def __next__(self):
        if not self.iter:
            raise StopIteration
        row = TreeModelRow(self.model, self.iter)
        self.iter = self.model.iter_next(self.iter)
        return row
    if GTK3:
        next = __next__

    def __iter__(self):
        return self