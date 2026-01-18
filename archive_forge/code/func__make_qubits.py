import itertools
import pytest
import numpy as np
import sympy
import cirq
def _make_qubits(n):
    return [cirq.NamedQubit(f'q{i}') for i in range(n)]