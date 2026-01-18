import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _find_immediate_dominators(self):

    def intersect(u, v):
        while u != v:
            while idx[u] < idx[v]:
                u = idom[u]
            while idx[u] > idx[v]:
                v = idom[v]
        return u
    entry = self._entry_point
    preds_table = self._preds
    order = self._find_postorder()
    idx = {e: i for i, e in enumerate(order)}
    idom = {entry: entry}
    order.pop()
    order.reverse()
    changed = True
    while changed:
        changed = False
        for u in order:
            new_idom = functools.reduce(intersect, (v for v in preds_table[u] if v in idom))
            if u not in idom or idom[u] != new_idom:
                idom[u] = new_idom
                changed = True
    return idom