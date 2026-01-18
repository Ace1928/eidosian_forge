from __future__ import annotations
import os
import pathlib
import sys
from functools import partial, update_wrapper
from inspect import cleandoc
from typing import IO, TYPE_CHECKING, Any, BinaryIO, ClassVar, TypeVar, overload
from trio._file_io import AsyncIOWrapper, wrap_file
from trio._util import final
from trio.to_thread import run_sync
def _wrap_method_path(fn: Callable[Concatenate[pathlib.Path, P], pathlib.Path]) -> Callable[Concatenate[PathT, P], Awaitable[PathT]]:

    @_wraps_async(fn)
    def wrapper(self: PathT, /, *args: P.args, **kwargs: P.kwargs) -> PathT:
        return self.__class__(fn(self._wrapped_cls(self), *args, **kwargs))
    return wrapper