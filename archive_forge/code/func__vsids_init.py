from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
def _vsids_init(self):
    """Initialize the data structures needed for the VSIDS heuristic."""
    self.lit_heap = []
    self.lit_scores = {}
    for var in range(1, len(self.variable_set)):
        self.lit_scores[var] = float(-self.occurrence_count[var])
        self.lit_scores[-var] = float(-self.occurrence_count[-var])
        heappush(self.lit_heap, (self.lit_scores[var], var))
        heappush(self.lit_heap, (self.lit_scores[-var], -var))