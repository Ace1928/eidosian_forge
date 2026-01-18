from __future__ import annotations
import math
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
@staticmethod
def _update_angles(angle1, angle2):
    """Calculate the new rotation angles according to Shende's decomposition."""
    return ((angle1 + angle2) / 2.0, (angle1 - angle2) / 2.0)