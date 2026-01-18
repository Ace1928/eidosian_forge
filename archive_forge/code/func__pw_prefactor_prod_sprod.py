from typing import Union
from functools import singledispatch
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.operation import Tensor
from .utils import is_pauli_word
from .conversion import pauli_sentence
@_pauli_word_prefactor.register(Prod)
@_pauli_word_prefactor.register(SProd)
def _pw_prefactor_prod_sprod(observable: Union[Prod, SProd]):
    ps = observable.pauli_rep
    if ps is not None and len(ps) == 1:
        return list(ps.values())[0]
    raise ValueError(f'Expected a valid Pauli word, got {observable}')