import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _find_dominator_tree(self):
    idom = self._idom
    domtree = _DictOfContainers(set)
    for u, v in idom.items():
        if u not in domtree:
            domtree[u] = set()
        if u != v:
            domtree[v].add(u)
    return domtree