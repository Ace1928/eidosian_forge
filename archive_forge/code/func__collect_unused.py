from typing import Dict, List, Optional, Set, TYPE_CHECKING
import abc
import collections
from cirq.devices import GridQubit
from cirq_google.line.placement import place_strategy
from cirq_google.line.placement.chip import chip_as_adjacency_list
from cirq_google.line.placement.sequence import GridQubitLineTuple
def _collect_unused(self, start: GridQubit, used: Set[GridQubit]) -> Set[GridQubit]:
    """Lists all the qubits that are reachable from given qubit.

        Args:
            start: The first qubit for which connectivity should be calculated.
                   Might be a member of used set.
            used: Already used qubits, which cannot be used during the
                  collection.

        Returns:
            Set of qubits that are reachable from starting qubit without
            traversing any of the used qubits.
        """

    def collect(n: GridQubit, visited: Set[GridQubit]):
        visited.add(n)
        for m in self._c_adj[n]:
            if m not in used and m not in visited:
                collect(m, visited)
    visited: Set[GridQubit] = set()
    collect(start, visited)
    return visited