import functools
import itertools
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import active_new_opmath
from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.wires import Wires
def _split_pauli_sentence(pl_sentence, max_size=15000):
    """Splits PauliSentences into smaller chunks of the size determined by the `max_size`.

    Args:
        pl_sentence (PauliSentence): PennyLane PauliSentence to be split
        max_size (int): Maximum size of each chunk

    Returns:
        Iterable consisting of smaller `PauliSentence` objects.
    """
    it, length = (iter(pl_sentence), len(pl_sentence))
    for _ in range(0, length, max_size):
        yield qml.pauli.PauliSentence({k: pl_sentence[k] for k in itertools.islice(it, max_size)})