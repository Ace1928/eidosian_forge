from __future__ import annotations
import itertools
import warnings
from types import FunctionType
from typing import Any, Callable
from typing_extensions import TypeAlias  # Python 3.10+
import jax.numpy as jnp
from jax import Array, lax
from jax._src import dtypes
from jax.typing import ArrayLike
from optree.ops import tree_flatten, tree_unflatten
from optree.typing import PyTreeSpec, PyTreeTypeVar
from optree.utils import safe_zip, total_order_sorted
def _ravel_leaves(leaves: list[ArrayLike]) -> tuple[Array, Callable[[Array], list[ArrayLike]]]:
    if not leaves:
        return (jnp.array([]), _unravel_empty)
    from_dtypes = tuple((dtypes.dtype(leaf) for leaf in leaves))
    to_dtype = dtypes.result_type(*from_dtypes)
    sizes = tuple((jnp.size(leaf) for leaf in leaves))
    shapes = tuple((jnp.shape(leaf) for leaf in leaves))
    indices = tuple(itertools.accumulate(sizes))
    if all((dt == to_dtype for dt in from_dtypes)):
        raveled = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
        return (raveled, HashablePartial(_unravel_leaves_single_dtype, indices, shapes))
    raveled = jnp.concatenate([jnp.ravel(lax.convert_element_type(leaf, to_dtype)) for leaf in leaves])
    return (raveled, HashablePartial(_unravel_leaves, indices, shapes, from_dtypes, to_dtype))