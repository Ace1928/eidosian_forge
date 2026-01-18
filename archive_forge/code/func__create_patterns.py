import numpy as np
from qiskit.circuit import QuantumCircuit
def _create_patterns(n):
    """Creating the patterns for the phase layers."""
    if n % 2 == 0:
        pat1 = _even_pattern1(n)
        pat2 = _even_pattern2(n)
    else:
        pat1 = _odd_pattern1(n)
        pat2 = _odd_pattern2(n)
    pats = {}
    layer = 0
    for i in range(n):
        pats[0, i] = (i, i)
    if n % 2 == 0:
        ind1 = (2 * n - 4) // 2
    else:
        ind1 = (2 * n - 4) // 2 - 1
    ind2 = 0
    while layer < n // 2:
        for i in range(n):
            pats[layer + 1, i] = (pat1[ind1 + i], pat2[ind2 + i])
        layer += 1
        ind1 -= 2
        ind2 += 2
    return pats