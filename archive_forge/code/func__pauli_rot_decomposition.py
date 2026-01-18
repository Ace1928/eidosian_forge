from typing import List
from warnings import warn
import numpy as np
from scipy.sparse.linalg import expm as sparse_expm
import pennylane as qml
from pennylane import math
from pennylane.math import expand_matrix
from pennylane.operation import (
from pennylane.ops.qubit import Hamiltonian
from pennylane.wires import Wires
from .sprod import SProd
from .sum import Sum
from .symbolicop import ScalarSymbolicOp
@staticmethod
def _pauli_rot_decomposition(base: Operator, coeff: complex):
    """Decomposes the exponential of a Pauli word into a PauliRot.

        Args:
            base (Operator): exponentiated operator
            coeff (complex): coefficient multiplying the exponentiated operator

        Returns:
            List[Operator]: list containing the PauliRot operator
        """
    coeff = math.real(2j * coeff)
    pauli_word = qml.pauli.pauli_word_to_string(base)
    if pauli_word == 'I' * base.num_wires:
        return []
    return [qml.PauliRot(theta=coeff, pauli_word=pauli_word, wires=base.wires)]