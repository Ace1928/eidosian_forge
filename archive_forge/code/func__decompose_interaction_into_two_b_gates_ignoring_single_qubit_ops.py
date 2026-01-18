from typing import Sequence, Union, List, Iterator, TYPE_CHECKING, Iterable, Optional
import numpy as np
from cirq import circuits, devices, linalg, ops, protocols
def _decompose_interaction_into_two_b_gates_ignoring_single_qubit_ops(qubits: Sequence['cirq.Qid'], kak_interaction_coefficients: Iterable[float]) -> List['cirq.Operation']:
    """Decompose using a minimal construction of two-qubit operations.

    References:
        Minimum construction of two-qubit quantum operations
        https://arxiv.org/abs/quant-ph/0312193
    """
    a, b = qubits
    x, y, z = kak_interaction_coefficients
    r = (np.sin(y) * np.cos(z)) ** 2
    r = max(0.0, r)
    if r > 0.499999999999:
        rb = [ops.ry(np.pi).on(b)]
    else:
        b1 = np.cos(y * 2) * np.cos(z * 2) / (1 - 2 * r)
        b1 = max(0.0, min(1, b1))
        b2 = np.arcsin(np.sqrt(b1))
        b3 = np.arccos(1 - 4 * r)
        rb = [ops.rz(-b2).on(b), ops.ry(-b3).on(b), ops.rz(-b2).on(b)]
    s = 1 if z < 0 else -1
    return [_B(a, b), ops.ry(s * 2 * x).on(a), *rb, _B(a, b)]