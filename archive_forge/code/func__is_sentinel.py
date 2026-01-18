from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
def _is_sentinel(self, lit, cls):
    """Check if a literal is a sentinel of a given clause.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> next(l._find_model())
        {1: True, 2: False, 3: False}
        >>> l._is_sentinel(2, 3)
        True
        >>> l._is_sentinel(-3, 1)
        False

        """
    return cls in self.sentinels[lit]