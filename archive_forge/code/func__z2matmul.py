from __future__ import annotations
import itertools
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.mixins import generate_apidocs, AdjointMixin
from qiskit.circuit import QuantumCircuit, Instruction
from .dihedral_circuits import _append_circuit
from .polynomial import SpecialPolynomial
def _z2matmul(self, left, right):
    """Compute product of two n x n z2 matrices."""
    prod = np.mod(np.dot(left, right), 2)
    return prod