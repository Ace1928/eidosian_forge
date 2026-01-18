from typing import Dict, List, Optional, Set, TYPE_CHECKING
import abc
import collections
from cirq.devices import GridQubit
from cirq_google.line.placement import place_strategy
from cirq_google.line.placement.chip import chip_as_adjacency_list
from cirq_google.line.placement.sequence import GridQubitLineTuple
def _choose_next_qubit(self, qubit: GridQubit, used: Set[GridQubit]) -> Optional[GridQubit]:
    analyzed: Set[GridQubit] = set()
    best = None
    best_size = None
    for m in self._c_adj[qubit]:
        if m not in used and m not in analyzed:
            reachable = self._collect_unused(m, used)
            analyzed.update(reachable)
            if best is None or best_size < len(reachable):
                best = m
                best_size = len(reachable)
    return best