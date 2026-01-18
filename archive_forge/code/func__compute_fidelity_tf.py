from functools import lru_cache
import autograd
import autoray as ar
import pennylane as qml
from .utils import cast
from .quantum import _check_density_matrix, _check_state_vector
@tf.custom_gradient
def _compute_fidelity_tf(dm0, dm1):
    fid = _compute_fidelity_vanilla(dm0, dm1)

    def vjp(grad_out):
        return _compute_fidelity_grad(dm0, dm1, grad_out)
    return (fid, vjp)