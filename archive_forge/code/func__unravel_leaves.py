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
def _unravel_leaves(indices: tuple[int, ...], shapes: tuple[tuple[int, ...]], from_dtypes: tuple[jnp.dtype, ...], to_dtype: jnp.dtype, flat: Array) -> list[Array]:
    if jnp.shape(flat) != (indices[-1],):
        raise ValueError(f'The unravel function expected an array of shape {(indices[-1],)}, got shape {jnp.shape(flat)}.')
    array_dtype = dtypes.dtype(flat)
    if array_dtype != to_dtype:
        raise ValueError(f'The unravel function expected an array of dtype {to_dtype}, got dtype {array_dtype}.')
    chunks = jnp.split(flat, indices[:-1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return [lax.convert_element_type(chunk.reshape(shape), dtype) for chunk, shape, dtype in safe_zip(chunks, shapes, from_dtypes)]