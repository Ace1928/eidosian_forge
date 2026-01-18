from functools import lru_cache
import autograd
import autoray as ar
import pennylane as qml
from .utils import cast
from .quantum import _check_density_matrix, _check_state_vector
def _compute_fidelity_vjp1(dm0, dm1, grad_out):
    """
    Compute the VJP of fidelity with respect to the second density matrix
    """
    return _compute_fidelity_vjp0(dm1, dm0, grad_out)