import itertools
import pytest
import numpy as np
import sympy
import cirq
def _all_interaction_gates(exponents=(1,)):
    for pauli0, invert0, pauli1, invert1, e in itertools.product(_paulis, _bools, _paulis, _bools, exponents):
        yield cirq.PauliInteractionGate(pauli0, invert0, pauli1, invert1, exponent=e)