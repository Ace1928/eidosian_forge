from __future__ import annotations
import types
from collections.abc import Hashable
from typing import (
from typing_extensions import NamedTuple  # Generic NamedTuple: Python 3.11+
from typing_extensions import OrderedDict  # Generic OrderedDict: Python 3.7.2+
from typing_extensions import Self  # Python 3.11+
from typing_extensions import TypeAlias  # Python 3.10+
from typing_extensions import Final, Protocol, runtime_checkable  # Python 3.8+
from optree import _C
from optree._C import PyTreeKind, PyTreeSpec
from optree._C import (
class PyTree(Generic[T]):
    """Generic PyTree type.

    >>> import torch
    >>> from optree.typing import PyTree
    >>> TensorTree = PyTree[torch.Tensor]
    >>> TensorTree  # doctest: +IGNORE_WHITESPACE
    typing.Union[torch.Tensor,
                 typing.Tuple[ForwardRef('PyTree[torch.Tensor]'), ...],
                 typing.List[ForwardRef('PyTree[torch.Tensor]')],
                 typing.Dict[typing.Any, ForwardRef('PyTree[torch.Tensor]')],
                 typing.Deque[ForwardRef('PyTree[torch.Tensor]')],
                 optree.typing.CustomTreeNode[ForwardRef('PyTree[torch.Tensor]')]]
    """

    @_tp_cache
    def __class_getitem__(cls, item: T | tuple[T] | tuple[T, str | None]) -> TypeAlias:
        """Instantiate a PyTree type with the given type."""
        if not isinstance(item, tuple):
            item = (item, None)
        if len(item) != 2:
            raise TypeError(f'{cls.__name__}[...] only supports a tuple of 2 items, a parameter and a string of type name, got {item!r}.')
        param, name = item
        if name is not None and (not isinstance(name, str)):
            raise TypeError(f'{cls.__name__}[...] only supports a tuple of 2 items, a parameter and a string of type name, got {item!r}.')
        if isinstance(param, _GenericAlias) and param.__origin__ is Union and hasattr(param, '__pytree_args__'):
            return param
        if name is not None:
            recurse_ref = ForwardRef(name)
        elif isinstance(param, TypeVar):
            recurse_ref = ForwardRef(f'{cls.__name__}[{param.__name__}]')
        elif isinstance(param, type):
            if param.__module__ == 'builtins':
                typename = param.__qualname__
            else:
                try:
                    typename = f'{param.__module__}.{param.__qualname__}'
                except AttributeError:
                    typename = f'{param.__module__}.{param.__name__}'
            recurse_ref = ForwardRef(f'{cls.__name__}[{typename}]')
        else:
            recurse_ref = ForwardRef(f'{cls.__name__}[{param!r}]')
        pytree_alias = Union[param, Tuple[recurse_ref, ...], List[recurse_ref], Dict[Any, recurse_ref], Deque[recurse_ref], CustomTreeNode[recurse_ref]]
        pytree_alias.__pytree_args__ = item
        return pytree_alias

    def __new__(cls) -> NoReturn:
        """Prohibit instantiation."""
        raise TypeError('Cannot instantiate special typing classes.')

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        """Prohibit subclassing."""
        raise TypeError('Cannot subclass special typing classes.')

    def __copy__(self) -> PyTree:
        """Immutable copy."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> PyTree:
        """Immutable copy."""
        return self