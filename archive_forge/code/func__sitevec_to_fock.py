import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def _sitevec_to_fock(det, format):
    """Convert a Slater determinant from site vector to occupation number vector representation.

    Args:
        det (list(int) or list(str)): determinant in site vector representation
        format (str): the format of the determinant

    Returns:
        tuple: tuple of integers representing binaries that correspond to occupation vectors in
            alpha and beta spin sectors

    **Example**

    >>> det = [1, 2, 1, 0, 0, 2]
    >>> _sitevec_to_fock(det, format = 'dmrg')
    (5, 34)

    >>> det = ["a", "b", "a", "0", "0", "b"]
    >>> _sitevec_to_fock(det, format = 'shci')
    (5, 34)
    """
    if format == 'dmrg':
        format_map = {0: '00', 1: '10', 2: '01', 3: '11'}
    elif format == 'shci':
        format_map = {'0': '00', 'a': '10', 'b': '01', '2': '11'}
    strab = [format_map[key] for key in det]
    stra = ''.join((i[0] for i in strab))
    strb = ''.join((i[1] for i in strab))
    inta = int(stra[::-1], 2)
    intb = int(strb[::-1], 2)
    return (inta, intb)