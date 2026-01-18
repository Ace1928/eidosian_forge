from __future__ import annotations
from math import pi
from qiskit.circuit import (
from . import (
Decomposition of CNOT gate.

    NOTE: this differs to CNOT by a global phase.
    The matrix returned is given by exp(1j * pi/4) * CNOT

    Args:
        plus_ry (bool): positive initial RY rotation
        plus_rxx (bool): positive RXX rotation.

    Returns:
        QuantumCircuit: The decomposed circuit for CNOT gate (up to
        global phase).
    