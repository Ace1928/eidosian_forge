from __future__ import unicode_literals
import typing
from collections import defaultdict, deque, namedtuple
from ._repr import make_repr
from .errors import FSError
from .path import abspath, combine, normpath
def _make_walker(self, *args, **kwargs):
    """Create a walker instance."""
    walker = self.walker_class(*args, **kwargs)
    return walker