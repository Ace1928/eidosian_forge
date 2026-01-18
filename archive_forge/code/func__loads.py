from __future__ import annotations
from functools import partialmethod
from importlib import import_module
from typing import IO, Any, AnyStr, Callable, Type
from ufoLib2.errors import ExtrasNotInstalledError
from ufoLib2.typing import PathLike, T
def _loads(cls: Type[T], s: str | bytes, *, __callback: Callable[..., T], **kwargs: Any) -> T:
    return __callback(s, cls, **kwargs)