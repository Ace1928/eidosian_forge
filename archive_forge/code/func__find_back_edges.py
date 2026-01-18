import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _find_back_edges(self, stats=None):
    """
        Find back edges.  An edge (src, dest) is a back edge if and
        only if *dest* dominates *src*.
        """
    if stats is not None:
        if not isinstance(stats, dict):
            raise TypeError(f'*stats* must be a dict; got {type(stats)}')
        stats.setdefault('iteration_count', 0)
    back_edges = set()
    stack = []
    succs_state = {}
    entry_point = self.entry_point()
    checked = set()

    def push_state(node):
        stack.append(node)
        succs_state[node] = [dest for dest in self._succs[node]]
    push_state(entry_point)
    iter_ct = 0
    while stack:
        iter_ct += 1
        tos = stack[-1]
        tos_succs = succs_state[tos]
        if tos_succs:
            cur_node = tos_succs.pop()
            if cur_node in stack:
                back_edges.add((tos, cur_node))
            elif cur_node not in checked:
                push_state(cur_node)
        else:
            stack.pop()
            checked.add(tos)
    if stats is not None:
        stats['iteration_count'] += iter_ct
    return back_edges