import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def _random_unitary_with_close_eigenvalues():
    U = cirq.testing.random_unitary(2)
    d = np.diag(np.exp([-0.2312j, -0.2312j]))
    return U @ d @ U.conj().T