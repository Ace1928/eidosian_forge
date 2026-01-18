from copy import deepcopy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import calc_inverse_matrix
from qiskit.synthesis.linear.linear_depth_lnn import _optimize_cx_circ_depth_5n_line
def _initialize_phase_schedule(mat_z):
    """
    Given a CZ layer (represented as an n*n CZ matrix Mz)
    Return a scheudle of phase gates implementing Mz in a SWAP-only netwrok
    (c.f. Alg 1, [2])
    """
    n = len(mat_z)
    phase_schedule = np.zeros((n, n), dtype=int)
    for i, j in zip(*np.where(mat_z)):
        if i >= j:
            continue
        phase_schedule[i, j] = 3
        phase_schedule[i, i] += 1
        phase_schedule[j, j] += 1
    return phase_schedule