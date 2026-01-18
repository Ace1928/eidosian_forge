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
def _unravel_empty(flat: Array) -> list[ArrayLike]:
    if jnp.shape(flat) != (0,):
        raise ValueError(f'The unravel function expected an array of shape {(0,)}, got shape {jnp.shape(flat)}.')
    return []