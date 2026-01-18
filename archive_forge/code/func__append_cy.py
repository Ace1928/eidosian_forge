from __future__ import annotations
import numpy as np
from qiskit.circuit import Barrier, Delay, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
def _append_cy(clifford, control, target):
    """Apply a CY gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        control (int): gate control qubit index.
        target (int): gate target qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford = _append_sdg(clifford, target)
    clifford = _append_cx(clifford, control, target)
    clifford = _append_s(clifford, target)
    return clifford