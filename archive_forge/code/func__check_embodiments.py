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
def _check_embodiments(self):
    """
        Checks that `self.embodiments` is populated with legal circuit embodiments: the key-value
        pair (angle, circuit) satisfies Operator(circuit) approx RXX(angle).to_matrix().
        """
    from qiskit.quantum_info.operators.measures import average_gate_fidelity
    for angle, embodiment in self.embodiments.items():
        actual = Operator(RXXGate(angle))
        purported = Operator(embodiment)
        if average_gate_fidelity(actual, purported) < 1 - EPSILON:
            raise QiskitError(f'RXX embodiment provided for angle {angle} disagrees with RXXGate({angle})')