from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING
from cirq import circuits, ops
def _get_qubit_mapping_first_and_last_moment(circuit: 'cirq.AbstractCircuit') -> Dict['cirq.Qid', Tuple[int, int]]:
    """Computes `(first_moment_idx, last_moment_idx)` tuple for each qubit in the input circuit.

    Args:
        circuit: An input cirq circuit to analyze.

    Returns:
        A dict mapping each qubit `q` in the input circuit to a tuple of integers
        `(first_moment_idx, last_moment_idx)` where
         - first_moment_idx: Index of leftmost moment containing an operation that acts on `q`.
         - last_moment_idx: Index of rightmost moment containing an operation that acts on `q`.
    """
    ret = {q: (len(circuit), 0) for q in circuit.all_qubits()}
    for i, moment in enumerate(circuit):
        for q in moment.qubits:
            ret[q] = (min(ret[q][0], i), max(ret[q][1], i))
    return ret