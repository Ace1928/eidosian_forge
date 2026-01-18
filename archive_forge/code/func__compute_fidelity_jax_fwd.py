from functools import lru_cache
import autograd
import autoray as ar
import pennylane as qml
from .utils import cast
from .quantum import _check_density_matrix, _check_state_vector
def _compute_fidelity_jax_fwd(dm0, dm1):
    fid = _compute_fidelity_jax(dm0, dm1)
    return (fid, (dm0, dm1))