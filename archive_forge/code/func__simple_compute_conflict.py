from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
def _simple_compute_conflict(self):
    """ Build a clause representing the fact that at least one decision made
        so far is wrong.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> next(l._find_model())
        {1: True, 2: False, 3: False}
        >>> l._simple_compute_conflict()
        [3]

        """
    return [-level.decision for level in self.levels[1:]]