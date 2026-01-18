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
def _wraps_async(wrapped: Callable[..., Any]) -> Callable[[Callable[P, T]], Callable[P, Awaitable[T]]]:

    def decorator(fn: Callable[P, T]) -> Callable[P, Awaitable[T]]:

        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await run_sync(partial(fn, *args, **kwargs))
        update_wrapper(wrapper, wrapped)
        assert wrapped.__doc__ is not None
        wrapper.__doc__ = f'Like :meth:`~{wrapped.__module__}.{wrapped.__qualname__}`, but async.\n\n{cleandoc(wrapped.__doc__)}\n'
        return wrapper
    return decorator