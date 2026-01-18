from functools import partial
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.typing import ResultBatch
from ..jacobian_products import _compute_jvps
from .jax import _NonPytreeWrapper
def _set_trainable_parameters_on_copy(tapes, params):
    """Copy a set of tapes with operations and set all trainable parameters"""
    return tuple((t.bind_new_parameters(a, t.trainable_params) for t, a in zip(tapes, params)))