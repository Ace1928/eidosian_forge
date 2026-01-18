from __future__ import annotations
import numpy as np
from qiskit.circuit import Barrier, Delay, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
def _append_w(clifford, qubit):
    """Apply a W gate to a Clifford.

    This is equivalent to two V gates.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    tmp = z.copy()
    z ^= x
    x[:] = tmp
    return clifford