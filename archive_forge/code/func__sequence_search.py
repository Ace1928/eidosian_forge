from typing import Dict, List, Optional, Set, TYPE_CHECKING
import abc
import collections
from cirq.devices import GridQubit
from cirq_google.line.placement import place_strategy
from cirq_google.line.placement.chip import chip_as_adjacency_list
from cirq_google.line.placement.sequence import GridQubitLineTuple
def _sequence_search(self, start: GridQubit, current: List[GridQubit]) -> List[GridQubit]:
    """Search for the continuous linear sequence from the given qubit.

        This method is called twice for the same starting qubit, so that
        sequences that begin and end on this qubit are searched for.

        Args:
            start: The first qubit, where search should be triggered from.
            current: Previously found linear sequence, which qubits are
                     forbidden to use during the search.

        Returns:
            Continuous linear sequence that begins with the starting qubit and
            does not contain any qubits from the current list.
        """
    used = set(current)
    seq = []
    n: Optional[GridQubit] = start
    while n is not None:
        seq.append(n)
        used.add(n)
        n = self._choose_next_qubit(n, used)
    return seq