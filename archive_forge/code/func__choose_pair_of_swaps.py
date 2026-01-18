from typing import Any, Dict, List, Optional, Set, Sequence, Tuple, TYPE_CHECKING
import itertools
import networkx as nx
from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.routing import mapping_manager, line_initial_mapper
@classmethod
def _choose_pair_of_swaps(cls, mm: mapping_manager.MappingManager, two_qubit_ops_ints: Sequence[Sequence[QidIntPair]], timestep: int, lookahead_radius: int) -> Optional[Tuple[QidIntPair, ...]]:
    """Computes cost function with pairs of candidate swaps that act on disjoint qubits."""
    pair_sigma = _disjoint_nc2_combinations(cls._initial_candidate_swaps(mm, two_qubit_ops_ints[timestep]))
    return cls._choose_optimal_swap(mm, two_qubit_ops_ints, timestep, lookahead_radius, pair_sigma)