from typing import Any, Dict, List, Optional, Set, Sequence, Tuple, TYPE_CHECKING
import itertools
import networkx as nx
from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.routing import mapping_manager, line_initial_mapper
@classmethod
def _initial_candidate_swaps(cls, mm: mapping_manager.MappingManager, two_qubit_ops: Sequence[QidIntPair]) -> List[QidIntPair]:
    """Finds all feasible SWAPs between qubits involved in 2-qubit operations."""
    physical_qubits = (mm.logical_to_physical[lq[i]] for lq in two_qubit_ops for i in range(2))
    physical_swaps = mm.induced_subgraph_int.edges(nbunch=physical_qubits)
    return [(mm.physical_to_logical[q1], mm.physical_to_logical[q2]) for q1, q2 in physical_swaps]