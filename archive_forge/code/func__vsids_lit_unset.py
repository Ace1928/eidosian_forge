from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
def _vsids_lit_unset(self, lit):
    """Handle the unsetting of a literal for the VSIDS heuristic.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> l.lit_heap
        [(-2.0, -3), (-2.0, 2), (-2.0, -2), (0.0, 1), (-2.0, 3), (0.0, -1)]

        >>> l._vsids_lit_unset(2)

        >>> l.lit_heap
        [(-2.0, -3), (-2.0, -2), (-2.0, -2), (-2.0, 2), (-2.0, 3), (0.0, -1),
        ...(-2.0, 2), (0.0, 1)]

        """
    var = abs(lit)
    heappush(self.lit_heap, (self.lit_scores[var], var))
    heappush(self.lit_heap, (self.lit_scores[-var], -var))