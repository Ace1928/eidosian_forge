import cmath
import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.analytical_decompositions.two_qubit_to_cz import (
from cirq.testing import random_two_qubit_circuit_with_czs
def _random_single_partial_cz_effect():
    return cirq.dot(cirq.kron(cirq.testing.random_unitary(2), cirq.testing.random_unitary(2)), np.diag([1, 1, 1, cmath.exp(2j * random.random() * np.pi)]), cirq.kron(cirq.testing.random_unitary(2), cirq.testing.random_unitary(2)))