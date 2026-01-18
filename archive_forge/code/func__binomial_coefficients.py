from __future__ import annotations
from itertools import product
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations
def _binomial_coefficients(n):
    """Return a dictionary of binomial coefficients

    Based-on/forked from sympy's binomial_coefficients() function [#]

    .. [#] https://github.com/sympy/sympy/blob/sympy-1.5.1/sympy/ntheory/multinomial.py
    """
    data = {(0, n): 1, (n, 0): 1}
    temp = 1
    for k in range(1, n // 2 + 1):
        temp = temp * (n - k + 1) // k
        data[k, n - k] = data[n - k, k] = temp
    return data