from functools import partial
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.typing import ResultBatch
from ..jacobian_products import _compute_jvps
from .jax import _NonPytreeWrapper
def _jax_dtype(m_type):
    if m_type == int:
        return jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
    if m_type == float:
        return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    if m_type == complex:
        return jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    return jnp.dtype(m_type)