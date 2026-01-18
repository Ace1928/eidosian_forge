from functools import partial
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.typing import ResultBatch
from ..jacobian_products import _compute_jvps
from .jax import _NonPytreeWrapper
def _vjp_bwd(tapes, execute_fn, jpc, device, params, dy):
    """Perform the backward pass of a vjp calculation, returning the vjp."""

    def wrapper(inner_params, inner_dy):
        new_tapes = _set_trainable_parameters_on_copy(tapes.vals, inner_params)
        return _to_jax(jpc.compute_vjp(new_tapes, inner_dy))
    vjp_shape = _pytree_shape_dtype_struct(params)
    return (jax.pure_callback(wrapper, vjp_shape, params, dy, vectorized=True),)