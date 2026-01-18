import functools
import itertools
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import active_new_opmath
from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.wires import Wires
def _tapered_op(params):
    """Applies the tapered operation for the specified parameter value whenever
        queing context is active, otherwise returns it as a list."""
    if qml.QueuingManager.recording():
        qml.QueuingManager.remove(operation)
        for coeff, op in zip(*gen_tapered.terms()):
            qml.exp(op, 1j * params * coeff)
    else:
        ops_tapered = []
        for coeff, op in zip(*gen_tapered.terms()):
            ops_tapered.append(qml.exp(op, 1j * params * coeff))
        return ops_tapered