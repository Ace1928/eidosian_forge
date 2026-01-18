import abc
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr
from cirq.qis import quantum_state_representation
from cirq.value import big_endian_int_to_digits, linear_dict, random_state
def _row_to_dense_pauli(self, i: int) -> 'cirq.DensePauliString':
    """Return a dense Pauli string for the given row in the tableau.

        Args:
            i: index of the row in the tableau.

        Returns:
            A DensePauliString representing the row. The length of the string
            is equal to the total number of qubits and each character
            represents the effective single Pauli operator on that qubit. The
            overall phase is captured in the coefficient.
        """
    from cirq.ops.dense_pauli_string import DensePauliString
    coefficient = -1 if self.rs[i] else 1
    pauli_mask = ''
    for k in range(self.n):
        if self.xs[i, k] & (not self.zs[i, k]):
            pauli_mask += 'X'
        elif (not self.xs[i, k]) & self.zs[i, k]:
            pauli_mask += 'Z'
        elif self.xs[i, k] & self.zs[i, k]:
            pauli_mask += 'Y'
        else:
            pauli_mask += 'I'
    return DensePauliString(pauli_mask, coefficient=coefficient)