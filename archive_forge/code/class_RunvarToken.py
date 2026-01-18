from __future__ import annotations
import enum
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, overload
from weakref import WeakKeyDictionary
from ._core._eventloop import get_async_backend
class RunvarToken(Generic[T]):
    __slots__ = ('_var', '_value', '_redeemed')

    def __init__(self, var: RunVar[T], value: T | Literal[_NoValueSet.NO_VALUE_SET]):
        self._var = var
        self._value: T | Literal[_NoValueSet.NO_VALUE_SET] = value
        self._redeemed = False