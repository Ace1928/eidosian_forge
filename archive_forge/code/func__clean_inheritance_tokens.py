import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _clean_inheritance_tokens(self):
    """create a new copy of this :class:`.Context`. with
        tokens related to inheritance state removed."""
    c = self._copy()
    x = c._data
    x.pop('self', None)
    x.pop('parent', None)
    x.pop('next', None)
    return c