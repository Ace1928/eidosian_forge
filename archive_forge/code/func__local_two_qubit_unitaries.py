import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def _local_two_qubit_unitaries(samples, random_state):
    kl_0 = np.array([cirq.testing.random_unitary(2, random_state=random_state) for _ in range(samples)])
    kl_1 = np.array([cirq.testing.random_unitary(2, random_state=random_state) for _ in range(samples)])
    return _vector_kron(kl_0, kl_1)