from typing import List, Union, Tuple
import numpy as np
import pennylane as qml
from pennylane.ops import Sum, Hamiltonian, SProd, Prod
from pennylane.measurements import (
from pennylane.typing import TensorLike
from .apply_operation import apply_operation
from .measure import flatten_state
def _apply_diagonalizing_gates(mps: List[SampleMeasurement], state: np.ndarray, is_state_batched: bool=False):
    if len(mps) == 1:
        diagonalizing_gates = mps[0].diagonalizing_gates()
    elif all((mp.obs for mp in mps)):
        diagonalizing_gates = qml.pauli.diagonalize_qwc_pauli_words([mp.obs for mp in mps])[0]
    else:
        diagonalizing_gates = []
    for op in diagonalizing_gates:
        state = apply_operation(op, state, is_state_batched=is_state_batched)
    return state