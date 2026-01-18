import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
from .clifford_decompose_bm import _decompose_clifford_1q
Set stabilizer.Z[qubit, i] to False for all i > qubit.

    Implemented by applying (reverse) CNOTS assumes qubit < num_qubits
    and _set_row_x_zero has been called first
    