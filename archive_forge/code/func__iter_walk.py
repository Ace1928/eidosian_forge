from __future__ import unicode_literals
import typing
from collections import defaultdict, deque, namedtuple
from ._repr import make_repr
from .errors import FSError
from .path import abspath, combine, normpath
def _iter_walk(self, fs, path, namespaces=None):
    """Get the walk generator."""
    if self.search == 'breadth':
        return self._walk_breadth(fs, path, namespaces=namespaces)
    else:
        return self._walk_depth(fs, path, namespaces=namespaces)