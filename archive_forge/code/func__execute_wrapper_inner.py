from functools import partial
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.typing import ResultBatch
from ..jacobian_products import _compute_jvps
from .jax import _NonPytreeWrapper
def _execute_wrapper_inner(params, tapes, execute_fn, _, device, is_vjp=False) -> ResultBatch:
    """
    Execute tapes using a pure-callback.
    """
    shape_dtype_structs = tuple((_result_shape_dtype_struct(t, device) for t in tapes.vals))
    _set_fn = _set_trainable_parameters_on_copy if is_vjp else _set_all_parameters_on_copy

    def pure_callback_wrapper(p):
        new_tapes = _set_fn(tapes.vals, p)
        res = tuple(_to_jax(execute_fn(new_tapes)))
        return jax.tree_map(lambda r, s: r.T if r.ndim > s.ndim else r, res, shape_dtype_structs)
    out = jax.pure_callback(pure_callback_wrapper, shape_dtype_structs, params, vectorized=True)
    return out