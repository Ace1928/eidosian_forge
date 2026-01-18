import numpy as np
import scipy.stats
import cirq
def _count_operations(operations):
    """Counts single-qubit, CNOT and CCNOT gates.

    Also validates that there are no other gates."""
    count_2x2 = 0
    count_cnot = 0
    count_ccnot = 0
    for operation in operations:
        u = cirq.unitary(operation)
        if u.shape == (2, 2):
            count_2x2 += 1
        elif u.shape == (4, 4):
            assert np.allclose(u, cirq.unitary(cirq.CNOT))
            count_cnot += 1
        elif u.shape == (8, 8):
            assert np.allclose(u, cirq.unitary(cirq.CCNOT))
            count_ccnot += 1
    return (count_2x2, count_cnot, count_ccnot)