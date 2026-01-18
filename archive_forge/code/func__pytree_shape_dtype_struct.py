from functools import partial
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.typing import ResultBatch
from ..jacobian_products import _compute_jvps
from .jax import _NonPytreeWrapper
def _pytree_shape_dtype_struct(pytree):
    """Creates a shape structure that matches the types and shapes for the provided pytree."""
    leaves, struct = jax.tree_util.tree_flatten(pytree)
    new_leaves = [jax.ShapeDtypeStruct(jnp.shape(l), l.dtype) for l in leaves]
    return jax.tree_util.tree_unflatten(struct, new_leaves)