from __future__ import annotations
import numpy as np
from qiskit.circuit import Barrier, Delay, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
def _append_v(clifford, qubit):
    """Apply a V gate to a Clifford.

    This is equivalent to an Sdg gate followed by a H gate.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    tmp = x.copy()
    x ^= z
    z[:] = tmp
    return clifford