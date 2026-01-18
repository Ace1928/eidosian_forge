from functools import lru_cache
import autograd
import autoray as ar
import pennylane as qml
from .utils import cast
from .quantum import _check_density_matrix, _check_state_vector
@lru_cache(maxsize=None)
def _register_tf_vjp():
    """
    Register the custom VJP for tensorflow
    """
    import tensorflow as tf

    @tf.custom_gradient
    def _compute_fidelity_tf(dm0, dm1):
        fid = _compute_fidelity_vanilla(dm0, dm1)

        def vjp(grad_out):
            return _compute_fidelity_grad(dm0, dm1, grad_out)
        return (fid, vjp)
    ar.register_function('tensorflow', 'compute_fidelity', _compute_fidelity_tf)