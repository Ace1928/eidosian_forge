from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
def _initialize_clauses(self, clauses):
    """Set up the clause data structures needed.

        For each clause, the following changes are made:
        - Unit clauses are queued for propagation right away.
        - Non-unit clauses have their first and last literals set as sentinels.
        - The number of clauses a literal appears in is computed.
        """
    self.clauses = [list(clause) for clause in clauses]
    for i, clause in enumerate(self.clauses):
        if 1 == len(clause):
            self._unit_prop_queue.append(clause[0])
            continue
        self.sentinels[clause[0]].add(i)
        self.sentinels[clause[-1]].add(i)
        for lit in clause:
            self.occurrence_count[lit] += 1