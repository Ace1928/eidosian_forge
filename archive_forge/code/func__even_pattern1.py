import numpy as np
from qiskit.circuit import QuantumCircuit
def _even_pattern1(n):
    """A pattern denoted by Pj in [1] for even number of qubits:
    [n-1, n-3, n-3, n-5, n-5, ..., 1, 1, 0, 0, 2, 2, ..., n-4, n-4, n-2]
    """
    pat = []
    pat.append(n - 1)
    for i in range((n - 2) // 2):
        pat.append(n - 2 * i - 3)
        pat.append(n - 2 * i - 3)
    for i in range((n - 2) // 2):
        pat.append(2 * i)
        pat.append(2 * i)
    pat.append(n - 2)
    return pat