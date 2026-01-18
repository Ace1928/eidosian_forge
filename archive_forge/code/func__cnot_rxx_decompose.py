from __future__ import annotations
from math import pi
from qiskit.circuit import (
from . import (
def _cnot_rxx_decompose(plus_ry: bool=True, plus_rxx: bool=True):
    """Decomposition of CNOT gate.

    NOTE: this differs to CNOT by a global phase.
    The matrix returned is given by exp(1j * pi/4) * CNOT

    Args:
        plus_ry (bool): positive initial RY rotation
        plus_rxx (bool): positive RXX rotation.

    Returns:
        QuantumCircuit: The decomposed circuit for CNOT gate (up to
        global phase).
    """
    if plus_ry:
        sgn_ry = 1
    else:
        sgn_ry = -1
    if plus_rxx:
        sgn_rxx = 1
    else:
        sgn_rxx = -1
    circuit = QuantumCircuit(2, global_phase=-sgn_ry * sgn_rxx * pi / 4)
    circuit.append(RYGate(sgn_ry * pi / 2), [0])
    circuit.append(RXXGate(sgn_rxx * pi / 2), [0, 1])
    circuit.append(RXGate(-sgn_rxx * pi / 2), [0])
    circuit.append(RXGate(-sgn_rxx * sgn_ry * pi / 2), [1])
    circuit.append(RYGate(-sgn_ry * pi / 2), [0])
    return circuit