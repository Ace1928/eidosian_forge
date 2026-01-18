from functools import lru_cache
import autograd
import autoray as ar
import pennylane as qml
from .utils import cast
from .quantum import _check_density_matrix, _check_state_vector
@autograd.extend.primitive
def _compute_fidelity_autograd(dm0, dm1):
    return _compute_fidelity_vanilla(dm0, dm1)