from __future__ import annotations
import numpy as np
from qiskit.circuit import Barrier, Delay, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
def _append_x(clifford, qubit):
    """Apply an X gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.phase ^= clifford.z[:, qubit]
    return clifford