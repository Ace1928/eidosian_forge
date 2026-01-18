import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def _wfdict_to_statevector(fcimatr_dict, norbs):
    """Convert a wavefunction in sparse dictionary format to a PennyLane statevector.

    In the sparse dictionary format, the keys ``(int_a, int_b)`` are integers whose binary
    representation shows the Fock occupation vector for alpha and beta electrons and values are the
    CI coefficients.

    Args:
        fcimatr_dict (dict[tuple(int,int),float]): the sparse dictionary format of a wavefunction
        norbs (int): number of molecular orbitals

    Returns:
        array: normalized state vector of length :math:`2^M`, where :math:`M` is the number of spin orbitals
    """
    statevector = np.zeros(2 ** (2 * norbs), dtype=complex)
    for (int_a, int_b), coeff in fcimatr_dict.items():
        bin_a = bin(int_a)[2:][::-1]
        bin_b = bin(int_b)[2:][::-1]
        bin_a += '0' * (norbs - len(bin_a))
        bin_b += '0' * (norbs - len(bin_b))
        bin_ab = ''.join((i + j for i, j in zip(bin_a, bin_b)))
        statevector[int(bin_ab, 2)] += coeff
    statevector = statevector / np.sqrt(np.sum(statevector ** 2))
    return statevector