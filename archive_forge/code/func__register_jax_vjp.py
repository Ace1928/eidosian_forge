from functools import lru_cache
import autograd
import autoray as ar
import pennylane as qml
from .utils import cast
from .quantum import _check_density_matrix, _check_state_vector
@lru_cache(maxsize=None)
def _register_jax_vjp():
    """
    Register the custom VJP for JAX
    """
    import jax

    @jax.custom_vjp
    def _compute_fidelity_jax(dm0, dm1):
        return _compute_fidelity_vanilla(dm0, dm1)

    def _compute_fidelity_jax_fwd(dm0, dm1):
        fid = _compute_fidelity_jax(dm0, dm1)
        return (fid, (dm0, dm1))

    def _compute_fidelity_jax_bwd(res, grad_out):
        dm0, dm1 = res
        return _compute_fidelity_grad(dm0, dm1, grad_out)
    _compute_fidelity_jax.defvjp(_compute_fidelity_jax_fwd, _compute_fidelity_jax_bwd)
    ar.register_function('jax', 'compute_fidelity', _compute_fidelity_jax)