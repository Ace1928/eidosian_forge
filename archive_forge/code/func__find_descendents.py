import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _find_descendents(self):
    descs = {}
    for node in reversed(self._topo_order):
        descs[node] = node_descs = set()
        for succ in self._succs[node]:
            if (node, succ) not in self._back_edges:
                node_descs.add(succ)
                node_descs.update(descs[succ])
    return descs