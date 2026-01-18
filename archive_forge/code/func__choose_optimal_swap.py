from typing import Any, Dict, List, Optional, Set, Sequence, Tuple, TYPE_CHECKING
import itertools
import networkx as nx
from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.routing import mapping_manager, line_initial_mapper
@classmethod
def _choose_optimal_swap(cls, mm: mapping_manager.MappingManager, two_qubit_ops_ints: Sequence[Sequence[QidIntPair]], timestep: int, lookahead_radius: int, sigma: Sequence[Tuple[QidIntPair, ...]]) -> Optional[Tuple[QidIntPair, ...]]:
    """Optionally returns the swap with minimum cost from a list of n-tuple candidate swaps.

        Computes a cost (as defined by the overridable function `_cost`) for each candidate swap
        in the current timestep. If there does not exist a unique list of swaps with minial cost,
        proceeds to the rank the subset of minimal swaps from the current timestep in the next
        timestep. Iterate this this looking ahead process up to the next `lookahead_radius`
        timesteps. If there still doesn't exist a unique swap with minial cost then returns None.
        """
    for s in range(timestep, min(lookahead_radius + timestep, len(two_qubit_ops_ints))):
        if len(sigma) <= 1:
            break
        costs = {}
        for swaps in sigma:
            costs[swaps] = cls._cost(mm, swaps, two_qubit_ops_ints[s])
        _, min_cost = min(costs.items(), key=lambda x: x[1])
        sigma = [swaps for swaps, cost in costs.items() if cost == min_cost]
    return None if len(sigma) > 1 and timestep + lookahead_radius <= len(two_qubit_ops_ints) else sigma[0]