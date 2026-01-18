import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def _validate_qubit_mapping(qubit_map: Mapping[TKey, int], pauli_qubits: Tuple[TKey, ...], num_state_qubits: int) -> None:
    """Validates that a qubit map is a valid mapping.

    This will enforce that all elements of `pauli_qubits` appear in `qubit_map`,
    and that the integers in `qubit_map` correspond to valid positions in a
    representation of a state over `num_state_qubits`.

    Args:
        qubit_map: A map from qubits to integers.
        pauli_qubits: The qubits that must be contained in `qubit_map`.
        num_state_qubits: The number of qubits over which a state is expressed.

    Raises:
        TypeError: If the qubit map is between the wrong types.
        ValueError: If the qubit maps is not complete or does not match with
            `num_state_qubits`.
    """
    if not isinstance(qubit_map, Mapping) or not all((isinstance(k, raw_types.Qid) and isinstance(v, int) for k, v in qubit_map.items())):
        raise TypeError("Input qubit map must be a valid mapping from Qubit ID's to integer indices.")
    if not set(qubit_map.keys()) >= set(pauli_qubits):
        raise ValueError("Input qubit map must be a complete mapping over all of this PauliString's qubits.")
    used_inds = [qubit_map[q] for q in pauli_qubits]
    if len(used_inds) != len(set(used_inds)) or not set(range(num_state_qubits)) >= set(sorted(used_inds)):
        raise ValueError(f'Input qubit map indices must be valid for a state over {num_state_qubits} qubits.')