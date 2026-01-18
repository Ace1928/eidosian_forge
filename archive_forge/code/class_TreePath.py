import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class TreePath(Gtk.TreePath):

    def __new__(cls, path=0):
        if isinstance(path, int):
            path = str(path)
        elif not isinstance(path, str):
            path = ':'.join((str(val) for val in path))
        if len(path) == 0:
            raise TypeError("could not parse subscript '%s' as a tree path" % path)
        try:
            return TreePath.new_from_string(path)
        except TypeError:
            raise TypeError("could not parse subscript '%s' as a tree path" % path)

    def __init__(self, *args, **kwargs):
        super(TreePath, self).__init__()

    def __str__(self):
        return self.to_string() or ''

    def __lt__(self, other):
        return other is not None and self.compare(other) < 0

    def __le__(self, other):
        return other is not None and self.compare(other) <= 0

    def __eq__(self, other):
        return other is not None and self.compare(other) == 0

    def __ne__(self, other):
        return other is None or self.compare(other) != 0

    def __gt__(self, other):
        return other is None or self.compare(other) > 0

    def __ge__(self, other):
        return other is None or self.compare(other) >= 0

    def __iter__(self):
        return iter(self.get_indices())

    def __len__(self):
        return self.get_depth()

    def __getitem__(self, index):
        return self.get_indices()[index]