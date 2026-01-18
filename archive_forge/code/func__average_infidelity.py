from __future__ import annotations
import heapq
import math
from operator import itemgetter
from typing import Callable
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXXGate, RZXGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.one_qubit.one_qubit_decompose import ONE_QUBIT_EULER_BASIS_GATES
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
from .circuits import apply_reflection, apply_shift, canonical_xx_circuit
from .utilities import EPSILON
from .polytopes import XXPolytope
def _average_infidelity(p, q):
    """
    Computes the infidelity distance between two points p, q expressed in positive canonical
    coordinates.
    """
    a0, b0, c0 = p
    a1, b1, c1 = q
    return 1 - 1 / 20 * (4 + 16 * (math.cos(a0 - a1) ** 2 * math.cos(b0 - b1) ** 2 * math.cos(c0 - c1) ** 2 + math.sin(a0 - a1) ** 2 * math.sin(b0 - b1) ** 2 * math.sin(c0 - c1) ** 2))