import numpy as np
from qiskit.circuit import QuantumCircuit
def _even_pattern2(n):
    """A pattern denoted by Pk in [1] for even number of qubits:
    [2, 2, 4, 4, ..., n-2, n-2, n-1, n-1, ..., 3, 3, 1, 1]
    """
    pat = []
    for i in range((n - 2) // 2):
        pat.append(2 * (i + 1))
        pat.append(2 * (i + 1))
    for i in range(n // 2):
        pat.append(n - 2 * i - 1)
        pat.append(n - 2 * i - 1)
    return pat