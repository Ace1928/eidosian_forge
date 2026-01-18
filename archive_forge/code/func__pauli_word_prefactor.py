from typing import Union
from functools import singledispatch
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.operation import Tensor
from .utils import is_pauli_word
from .conversion import pauli_sentence
@singledispatch
def _pauli_word_prefactor(observable):
    """Private wrapper function for pauli_word_prefactor."""
    raise ValueError(f'Expected a valid Pauli word, got {observable}')