from typing import Callable, List, Optional, Tuple, Set, Any, TYPE_CHECKING
import numpy as np
import cirq
from cirq_google.line.placement import place_strategy, optimization
from cirq_google.line.placement.chip import above, right_of, chip_as_adjacency_list, EDGE
from cirq_google.line.placement.sequence import GridQubitLineTuple, LineSequence
def _quadratic_sum_cost(self, state: _STATE) -> float:
    """Cost function that sums squares of lengths of sequences.

        Args:
          state: Search state, not mutated.

        Returns:
          Cost which is minus the normalized quadratic sum of each linear
          sequence section in the state. This promotes single, long linear
          sequence solutions and converges to number -1. The solution with a
          lowest cost consists of every node being a single sequence and is
          always less than 0.
        """
    cost = 0.0
    total_len = float(len(self._c))
    seqs, _ = state
    for seq in seqs:
        cost += (len(seq) / total_len) ** 2
    return -cost