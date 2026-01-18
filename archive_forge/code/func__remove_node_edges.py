import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _remove_node_edges(self, node):
    for succ in self._succs.pop(node, ()):
        self._preds[succ].remove(node)
        del self._edge_data[node, succ]
    for pred in self._preds.pop(node, ()):
        self._succs[pred].remove(node)
        del self._edge_data[pred, node]