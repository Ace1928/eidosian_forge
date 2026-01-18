from typing import Any, Dict, List, Optional, Set, Sequence, Tuple, TYPE_CHECKING
import itertools
import networkx as nx
from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.routing import mapping_manager, line_initial_mapper
def _disjoint_nc2_combinations(qubit_pairs: Sequence[QidIntPair]) -> List[Tuple[QidIntPair, QidIntPair]]:
    """Gets disjoint pair combinations of qubits pairs.

    For example:

        >>> q = [*range(5)]
        >>> disjoint_swaps = cirq.transformers.routing.route_circuit_cqc._disjoint_nc2_combinations(
        ...     [(q[0], q[1]), (q[2], q[3]), (q[1], q[4])]
        ... )
        >>> disjoint_swaps == [((q[0], q[1]), (q[2], q[3])), ((q[2], q[3]), (q[1], q[4]))]
        True

    Args:
        qubit_pairs: list of qubit pairs to be combined.

    Returns:
        All 2-combinations between qubit pairs that are disjoint.
    """
    return [pair for pair in itertools.combinations(qubit_pairs, 2) if set(pair[0]).isdisjoint(pair[1])]