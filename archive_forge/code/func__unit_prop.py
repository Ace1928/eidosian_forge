from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
def _unit_prop(self):
    """Perform unit propagation on the current theory."""
    result = len(self._unit_prop_queue) > 0
    while self._unit_prop_queue:
        next_lit = self._unit_prop_queue.pop()
        if -next_lit in self.var_settings:
            self.is_unsatisfied = True
            self._unit_prop_queue = []
            return False
        else:
            self._assign_literal(next_lit)
    return result