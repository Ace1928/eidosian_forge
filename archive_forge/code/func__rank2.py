from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
def _rank2(a, b, c, d):
    """Return rank of 2x2 boolean matrix."""
    if a & d ^ b & c:
        return 2
    if a or b or c or d:
        return 1
    return 0