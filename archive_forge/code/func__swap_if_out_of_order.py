from typing import Any, Dict, Iterable, Sequence, Tuple, TYPE_CHECKING
from cirq import protocols, value
from cirq.ops import raw_types, swap_gates
def _swap_if_out_of_order(idx: int) -> Iterable['cirq.Operation']:
    nonlocal is_sorted
    if self._permutation[qubit_ids[idx]] > self._permutation[qubit_ids[idx + 1]]:
        yield swap_gates.SWAP(qubits[idx], qubits[idx + 1])
        qubit_ids[idx + 1], qubit_ids[idx] = (qubit_ids[idx], qubit_ids[idx + 1])
        is_sorted = False