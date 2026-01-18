from __future__ import annotations
from typing import Any
import copy
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap
def _build_edges(self) -> set[tuple[int, int]]:
    """Build the possible edges that the swap strategy accommodates."""
    possible_edges = set()
    for swap_layer_idx in range(len(self) + 1):
        for edge in self.swapped_coupling_map(swap_layer_idx).get_edges():
            possible_edges.add(edge)
            possible_edges.add(edge[::-1])
    return possible_edges