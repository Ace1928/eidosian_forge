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
class PyTreeTypeVar:
    """Type variable for PyTree.

    >>> import torch
    >>> from optree.typing import PyTreeTypeVar
    >>> TensorTree = PyTreeTypeVar('TensorTree', torch.Tensor)
    >>> TensorTree  # doctest: +IGNORE_WHITESPACE
    typing.Union[torch.Tensor,
                 typing.Tuple[ForwardRef('TensorTree'), ...],
                 typing.List[ForwardRef('TensorTree')],
                 typing.Dict[typing.Any, ForwardRef('TensorTree')],
                 typing.Deque[ForwardRef('TensorTree')],
                 optree.typing.CustomTreeNode[ForwardRef('TensorTree')]]
    """

    @_tp_cache
    def __new__(cls, name: str, param: type) -> TypeAlias:
        """Instantiate a PyTree type variable with the given name and parameter."""
        if not isinstance(name, str):
            raise TypeError(f'{cls.__name__} only supports a string of type name, got {name!r}.')
        return PyTree[param, name]

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        """Prohibit subclassing."""
        raise TypeError('Cannot subclass special typing classes.')

    def __copy__(self) -> TypeAlias:
        """Immutable copy."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> TypeAlias:
        """Immutable copy."""
        return self