from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
@property
def _current_level(self):
    """The current decision level data structure

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{1}, {2}], {1, 2}, set())
        >>> next(l._find_model())
        {1: True, 2: True}
        >>> l._current_level.decision
        0
        >>> l._current_level.flipped
        False
        >>> l._current_level.var_settings
        {1, 2}

        """
    return self.levels[-1]