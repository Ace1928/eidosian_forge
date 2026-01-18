from __future__ import annotations
import copy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
def _lwr_cnot_synth(state, section_size):
    """
    This function is a helper function of the algorithm for optimal synthesis
    of linear reversible circuits (the Patel–Markov–Hayes algorithm). It works
    like gaussian elimination, except that it works a lot faster, and requires
    fewer steps (and therefore fewer CNOTs). It takes the matrix "state" and
    splits it into sections of size section_size. Then it eliminates all non-zero
    sub-rows within each section, which are the same as a non-zero sub-row
    above. Once this has been done, it continues with normal gaussian elimination.
    The benefit is that with small section sizes (m), most of the sub-rows will
    be cleared in the first step, resulting in a factor m fewer row row operations
    during Gaussian elimination.

    The algorithm is described in detail in the following paper
    "Optimal synthesis of linear reversible circuits."
    Patel, Ketan N., Igor L. Markov, and John P. Hayes.
    Quantum Information & Computation 8.3 (2008): 282-294.

    Note:
    This implementation tweaks the Patel, Markov, and Hayes algorithm by adding
    a "back reduce" which adds rows below the pivot row with a high degree of
    overlap back to it. The intuition is to avoid a high-weight pivot row
    increasing the weight of lower rows.

    Args:
        state (ndarray): n x n matrix, describing a linear quantum circuit
        section_size (int): the section size the matrix columns are divided into

    Returns:
        numpy.matrix: n by n matrix, describing the state of the output circuit
        list: a k by 2 list of C-NOT operations that need to be applied
    """
    circuit = []
    num_qubits = state.shape[0]
    cutoff = 1
    for sec in range(1, int(np.floor(num_qubits / section_size) + 1)):
        patt = {}
        for row in range((sec - 1) * section_size, num_qubits):
            sub_row_patt = copy.deepcopy(state[row, (sec - 1) * section_size:sec * section_size])
            if np.sum(sub_row_patt) == 0:
                continue
            if str(sub_row_patt) not in patt:
                patt[str(sub_row_patt)] = row
            else:
                state[row, :] ^= state[patt[str(sub_row_patt)], :]
                circuit.append([patt[str(sub_row_patt)], row])
        for col in range((sec - 1) * section_size, sec * section_size):
            diag_one = 1
            if state[col, col] == 0:
                diag_one = 0
            for row in range(col + 1, num_qubits):
                if state[row, col] == 1:
                    if diag_one == 0:
                        state[col, :] ^= state[row, :]
                        circuit.append([row, col])
                        diag_one = 1
                    state[row, :] ^= state[col, :]
                    circuit.append([col, row])
                if sum(state[col, :] & state[row, :]) > cutoff:
                    state[col, :] ^= state[row, :]
                    circuit.append([row, col])
    return [state, circuit]