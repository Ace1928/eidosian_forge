import cmath
import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.analytical_decompositions.two_qubit_to_cz import (
from cirq.testing import random_two_qubit_circuit_with_czs
def _random_double_full_cz_effect():
    return cirq.dot(cirq.kron(cirq.testing.random_unitary(2), cirq.testing.random_unitary(2)), cirq.unitary(cirq.CZ), cirq.kron(cirq.testing.random_unitary(2), cirq.testing.random_unitary(2)), cirq.unitary(cirq.CZ), cirq.kron(cirq.testing.random_unitary(2), cirq.testing.random_unitary(2)))