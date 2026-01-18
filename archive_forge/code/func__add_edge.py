import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _add_edge(self, from_, to, data=None):
    self._preds[to].add(from_)
    self._succs[from_].add(to)
    self._edge_data[from_, to] = data