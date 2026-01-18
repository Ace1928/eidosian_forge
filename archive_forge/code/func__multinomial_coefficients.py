from __future__ import annotations
from itertools import product
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations
def _multinomial_coefficients(m, n):
    """Return an iterator of multinomial coefficients

    Based-on/forked from sympy's multinomial_coefficients() function [#]

    .. [#] https://github.com/sympy/sympy/blob/sympy-1.5.1/sympy/ntheory/multinomial.py
    """
    if not m:
        if n:
            return {}
        return {(): 1}
    if m == 2:
        return _binomial_coefficients(n)
    if m >= 2 * n and n > 1:
        return dict(_large_coefficients_iter(m, n))
    if n:
        j = 0
    else:
        j = m
    temp = [n] + [0] * (m - 1)
    res = {tuple(temp): 1}
    while j < m - 1:
        temp_j = temp[j]
        if j:
            temp[j] = 0
            temp[0] = temp_j
        if temp_j > 1:
            temp[j + 1] += 1
            j = 0
            start = 1
            v = 0
        else:
            j += 1
            start = j + 1
            v = res[tuple(temp)]
            temp[j] += 1
        for k in range(start, m):
            if temp[k]:
                temp[k] -= 1
                v += res[tuple(temp)]
                temp[k] += 1
        temp[0] -= 1
        res[tuple(temp)] = v * temp_j // (n - temp[0])
    return res