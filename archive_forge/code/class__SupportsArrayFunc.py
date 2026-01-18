from __future__ import annotations
import sys
from collections.abc import Collection, Callable, Sequence
from typing import Any, Protocol, Union, TypeVar, runtime_checkable
from numpy import (
from ._nested_sequence import _NestedSequence
@runtime_checkable
class _SupportsArrayFunc(Protocol):
    """A protocol class representing `~class.__array_function__`."""

    def __array_function__(self, func: Callable[..., Any], types: Collection[type[Any]], args: tuple[Any, ...], kwargs: dict[str, Any]) -> object:
        ...