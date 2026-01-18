import random
import numpy as np
import pytest
import cirq
def _random_double_MS_effect():
    t1 = random.random()
    s1 = np.sin(t1)
    c1 = np.cos(t1)
    t2 = random.random()
    s2 = np.sin(t2)
    c2 = np.cos(t2)
    return cirq.dot(cirq.kron(cirq.testing.random_unitary(2), cirq.testing.random_unitary(2)), np.array([[c1, 0, 0, -1j * s1], [0, c1, -1j * s1, 0], [0, -1j * s1, c1, 0], [-1j * s1, 0, 0, c1]]), cirq.kron(cirq.testing.random_unitary(2), cirq.testing.random_unitary(2)), np.array([[c2, 0, 0, -1j * s2], [0, c2, -1j * s2, 0], [0, -1j * s2, c2, 0], [-1j * s2, 0, 0, c2]]), cirq.kron(cirq.testing.random_unitary(2), cirq.testing.random_unitary(2)))