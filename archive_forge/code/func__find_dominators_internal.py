import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _find_dominators_internal(self, post=False):
    if post:
        entries = set(self._exit_points)
        preds_table = self._succs
        succs_table = self._preds
    else:
        entries = set([self._entry_point])
        preds_table = self._preds
        succs_table = self._succs
    if not entries:
        raise RuntimeError('no entry points: dominator algorithm cannot be seeded')
    doms = {}
    for e in entries:
        doms[e] = set([e])
    todo = []
    for n in self._nodes:
        if n not in entries:
            doms[n] = set(self._nodes)
            todo.append(n)
    while todo:
        n = todo.pop()
        if n in entries:
            continue
        new_doms = set([n])
        preds = preds_table[n]
        if preds:
            new_doms |= functools.reduce(set.intersection, [doms[p] for p in preds])
        if new_doms != doms[n]:
            assert len(new_doms) < len(doms[n])
            doms[n] = new_doms
            todo.extend(succs_table[n])
    return doms