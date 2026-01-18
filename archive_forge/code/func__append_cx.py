from __future__ import annotations
import numpy as np
from qiskit.circuit import Barrier, Delay, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
def _append_cx(clifford, control, target):
    """Apply a CX gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        control (int): gate control qubit index.
        target (int): gate target qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x0 = clifford.x[:, control]
    z0 = clifford.z[:, control]
    x1 = clifford.x[:, target]
    z1 = clifford.z[:, target]
    clifford.phase ^= (x1 ^ z0 ^ True) & z1 & x0
    x1 ^= x0
    z0 ^= z1
    return clifford