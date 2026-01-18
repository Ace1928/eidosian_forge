from __future__ import annotations
from itertools import product
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations
def _large_coefficients_iter(m, n):
    """Return an iterator of multinomial coefficients

    Based-on/forked from sympy's multinomial_coefficients_iterator() function [#]

    .. [#] https://github.com/sympy/sympy/blob/sympy-1.5.1/sympy/ntheory/multinomial.py
    """
    if m < 2 * n or n == 1:
        coefficients = _multinomial_coefficients(m, n)
        for key, value in coefficients.items():
            yield (key, value)
    else:
        coefficients = _multinomial_coefficients(n, n)
        coefficients_dict = {}
        for key, value in coefficients.items():
            coefficients_dict[tuple(filter(None, key))] = value
        coefficients = coefficients_dict
        temp = [n] + [0] * (m - 1)
        temp_a = tuple(temp)
        b = tuple(filter(None, temp_a))
        yield (temp_a, coefficients[b])
        if n:
            j = 0
        else:
            j = m
        while j < m - 1:
            temp_j = temp[j]
            if j:
                temp[j] = 0
                temp[0] = temp_j
            if temp_j > 1:
                temp[j + 1] += 1
                j = 0
            else:
                j += 1
                temp[j] += 1
            temp[0] -= 1
            temp_a = tuple(temp)
            b = tuple(filter(None, temp_a))
            yield (temp_a, coefficients[b])