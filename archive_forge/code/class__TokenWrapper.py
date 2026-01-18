from __future__ import annotations
import enum
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, overload
from weakref import WeakKeyDictionary
from ._core._eventloop import get_async_backend
@dataclass(frozen=True)
class _TokenWrapper:
    __slots__ = ('_token', '__weakref__')
    _token: object