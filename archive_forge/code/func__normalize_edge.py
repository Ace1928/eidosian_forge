from typing import Callable, List, Optional, Tuple, Set, Any, TYPE_CHECKING
import numpy as np
import cirq
from cirq_google.line.placement import place_strategy, optimization
from cirq_google.line.placement.chip import above, right_of, chip_as_adjacency_list, EDGE
from cirq_google.line.placement.sequence import GridQubitLineTuple, LineSequence
def _normalize_edge(self, edge: EDGE) -> EDGE:
    """Gives unique representative of the edge.

        Two edges are equivalent if they form an edge between the same nodes.
        This method returns representative of this edge which can be compared
        using equality operator later.

        Args:
          edge: Edge to normalize.

        Returns:
          Normalized edge with lexicographically lower node on the first
          position.
        """

    def lower(n: cirq.GridQubit, m: cirq.GridQubit) -> bool:
        return n.row < m.row or (n.row == m.row and n.col < m.col)
    n1, n2 = edge
    return (n1, n2) if lower(n1, n2) else (n2, n1)