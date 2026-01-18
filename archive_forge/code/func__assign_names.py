from collections import deque
from numba.core import types, cgutils
def _assign_names(self, val_or_nested, name, depth=()):
    if isinstance(val_or_nested, (tuple, list)):
        for pos, aval in enumerate(val_or_nested):
            self._assign_names(aval, name, depth=depth + (pos,))
    else:
        postfix = '.'.join(map(str, depth))
        parts = [name, postfix]
        val_or_nested.name = '.'.join(filter(bool, parts))