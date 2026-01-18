import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, Pauli
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
Calculate a decoupling operator D:
    D^{-1} * Ox * D = x1
    D^{-1} * Oz * D = z1
    and reduce the clifford such that it will act trivially on min_qubit
    