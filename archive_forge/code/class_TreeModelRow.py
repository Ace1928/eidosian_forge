import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class TreeModelRow(object):

    def __init__(self, model, iter_or_path):
        if not isinstance(model, Gtk.TreeModel):
            raise TypeError('expected Gtk.TreeModel, %s found' % type(model).__name__)
        self.model = model
        if isinstance(iter_or_path, Gtk.TreePath):
            self.iter = model.get_iter(iter_or_path)
        elif isinstance(iter_or_path, Gtk.TreeIter):
            self.iter = iter_or_path
        else:
            raise TypeError('expected Gtk.TreeIter or Gtk.TreePath, %s found' % type(iter_or_path).__name__)

    @property
    def path(self):
        return self.model.get_path(self.iter)

    @property
    def next(self):
        return self.get_next()

    @property
    def previous(self):
        return self.get_previous()

    @property
    def parent(self):
        return self.get_parent()

    def get_next(self):
        next_iter = self.model.iter_next(self.iter)
        if next_iter:
            return TreeModelRow(self.model, next_iter)

    def get_previous(self):
        prev_iter = self.model.iter_previous(self.iter)
        if prev_iter:
            return TreeModelRow(self.model, prev_iter)

    def get_parent(self):
        parent_iter = self.model.iter_parent(self.iter)
        if parent_iter:
            return TreeModelRow(self.model, parent_iter)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key >= self.model.get_n_columns():
                raise IndexError('column index is out of bounds: %d' % key)
            elif key < 0:
                key = self._convert_negative_index(key)
            return self.model.get_value(self.iter, key)
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.model.get_n_columns())
            alist = []
            for i in range(start, stop, step):
                alist.append(self.model.get_value(self.iter, i))
            return alist
        elif isinstance(key, tuple):
            return [self[k] for k in key]
        else:
            raise TypeError('indices must be integers, slice or tuple, not %s' % type(key).__name__)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if key >= self.model.get_n_columns():
                raise IndexError('column index is out of bounds: %d' % key)
            elif key < 0:
                key = self._convert_negative_index(key)
            self.model.set_value(self.iter, key, value)
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.model.get_n_columns())
            indexList = range(start, stop, step)
            if len(indexList) != len(value):
                raise ValueError('attempt to assign sequence of size %d to slice of size %d' % (len(value), len(indexList)))
            for i, v in enumerate(indexList):
                self.model.set_value(self.iter, v, value[i])
        elif isinstance(key, tuple):
            if len(key) != len(value):
                raise ValueError('attempt to assign sequence of size %d to sequence of size %d' % (len(value), len(key)))
            for k, v in zip(key, value):
                self[k] = v
        else:
            raise TypeError('indices must be an integer, slice or tuple, not %s' % type(key).__name__)

    def _convert_negative_index(self, index):
        new_index = self.model.get_n_columns() + index
        if new_index < 0:
            raise IndexError('column index is out of bounds: %d' % index)
        return new_index

    def iterchildren(self):
        child_iter = self.model.iter_children(self.iter)
        return TreeModelRowIter(self.model, child_iter)