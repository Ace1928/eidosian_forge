import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, Pauli
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
def _compute_greedy_cost(list_pairs):
    """Compute the CNOT cost of one step of the algorithm"""
    A_num = 0
    B_num = 0
    C_num = 0
    D_num = 0
    for pair in list_pairs:
        if pair in A_class:
            A_num += 1
        elif pair in B_class:
            B_num += 1
        elif pair in C_class:
            C_num += 1
        elif pair in D_class:
            D_num += 1
    if A_num % 2 == 0:
        raise QiskitError('Symplectic Gaussian elimination fails.')
    cost = 3 * (A_num - 1) / 2 + (B_num + 1) * (B_num > 0) + C_num + D_num
    if list_pairs[0] not in A_class:
        cost += 3
    return cost