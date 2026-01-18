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
class HashablePartial:
    """A hashable version of :class:`functools.partial`."""
    func: FunctionType
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, func: FunctionType | HashablePartial, *args: Any, **kwargs: Any) -> None:
        """Construct a :class:`HashablePartial` instance."""
        if not callable(func):
            raise TypeError(f'Expected a callable, got {func!r}.')
        if isinstance(func, HashablePartial):
            self.func = func.func
            self.args = func.args + args
            self.kwargs = {**func.kwargs, **kwargs}
        elif isinstance(func, FunctionType):
            self.func = func
            self.args = args
            self.kwargs = kwargs
        else:
            raise TypeError(f'Expected a function, got {func!r}.')

    def __eq__(self, other: object) -> bool:
        return type(other) is HashablePartial and self.func.__code__ == other.func.__code__ and (self.args == other.args) and (self.kwargs == other.kwargs)

    def __hash__(self) -> int:
        return hash((self.func.__code__, self.args, tuple(total_order_sorted(self.kwargs.items(), key=lambda kv: kv[0]))))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*self.args, *args, **self.kwargs, **kwargs)