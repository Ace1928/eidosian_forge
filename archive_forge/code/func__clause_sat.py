from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
def _clause_sat(self, cls):
    """Check if a clause is satisfied by the current variable setting.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{1}, {-1}], {1}, set())
        >>> try:
        ...     next(l._find_model())
        ... except StopIteration:
        ...     pass
        >>> l._clause_sat(0)
        False
        >>> l._clause_sat(1)
        True

        """
    for lit in self.clauses[cls]:
        if lit in self.var_settings:
            return True
    return False