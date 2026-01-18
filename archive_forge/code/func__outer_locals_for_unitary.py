from functools import reduce
from typing import List, NamedTuple, Sequence, Tuple
from dataclasses import dataclass
import numpy as np
import cirq
from cirq import value
from cirq._compat import proper_repr, proper_eq
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
def _outer_locals_for_unitary(target: np.ndarray, base: np.ndarray) -> Tuple[_SingleQubitGatePair, _SingleQubitGatePair, np.ndarray]:
    """Local unitaries mapping between locally equivalent 2-local unitaries.

    Finds the left and right 1-local unitaries kL, kR such that

    U_target = kL @ U_base @ kR

    Args:
        target: The unitary to which we want to map.
        base: The base unitary which maps to target.

    Returns:
        kR: The right 1-local unitaries in the equation above, expressed as
            2-tuples of (2x2) single qubit unitaries.
        kL: The left 1-local unitaries in the equation above, expressed as
            2-tuples of (2x2) single qubit unitaries.
        actual: The outcome of kL @ base @ kR
    """
    target_decomp = cirq.kak_decomposition(target)
    base_decomp = cirq.kak_decomposition(base)
    kLt0, kLt1 = target_decomp.single_qubit_operations_after
    kLb0, kLb1 = base_decomp.single_qubit_operations_after
    kL = (kLt0 @ kLb0.conj().T, kLt1 @ kLb1.conj().T)
    kRt0, kRt1 = target_decomp.single_qubit_operations_before
    kRb0, kRb1 = base_decomp.single_qubit_operations_before
    kR = (kRb0.conj().T @ kRt0, kRb1.conj().T @ kRt1)
    actual = np.kron(*kL) @ base
    actual = actual @ np.kron(*kR)
    actual *= np.conj(target_decomp.global_phase)
    return (kR, kL, actual)