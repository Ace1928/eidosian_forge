from typing import Dict, List, Tuple, TYPE_CHECKING
import cirq
Gives adjacency list representation of a chip.

    The adjacency list is constructed in order of above, left_of, below and
    right_of consecutively.

    Args:
        device: Chip to be converted.

    Returns:
        Map from nodes to list of qubits which represent all the neighbours of
        given qubit.
    