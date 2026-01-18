from __future__ import annotations
import enum
import types
from typing import TYPE_CHECKING, Any, Callable, NoReturn
import attrs
import outcome
from . import _run
@types.coroutine
def _async_yield(obj: Any) -> Any:
    return (yield obj)