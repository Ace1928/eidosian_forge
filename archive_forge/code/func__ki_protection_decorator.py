from __future__ import annotations
import inspect
import signal
import sys
from functools import wraps
from typing import TYPE_CHECKING, Final, Protocol, TypeVar
import attrs
from .._util import is_main_thread
def _ki_protection_decorator(enabled: bool) -> Callable[[Callable[ArgsT, RetT]], Callable[ArgsT, RetT]]:

    def decorator(fn: Callable[ArgsT, RetT]) -> Callable[ArgsT, RetT]:
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            def wrapper(*args: ArgsT.args, **kwargs: ArgsT.kwargs) -> RetT:
                coro = fn(*args, **kwargs)
                coro.cr_frame.f_locals[LOCALS_KEY_KI_PROTECTION_ENABLED] = enabled
                return coro
            return wrapper
        elif inspect.isgeneratorfunction(fn):

            @wraps(fn)
            def wrapper(*args: ArgsT.args, **kwargs: ArgsT.kwargs) -> RetT:
                gen = fn(*args, **kwargs)
                gen.gi_frame.f_locals[LOCALS_KEY_KI_PROTECTION_ENABLED] = enabled
                return gen
            return wrapper
        elif inspect.isasyncgenfunction(fn) or legacy_isasyncgenfunction(fn):

            @wraps(fn)
            def wrapper(*args: ArgsT.args, **kwargs: ArgsT.kwargs) -> RetT:
                agen = fn(*args, **kwargs)
                agen.ag_frame.f_locals[LOCALS_KEY_KI_PROTECTION_ENABLED] = enabled
                return agen
            return wrapper
        else:

            @wraps(fn)
            def wrapper(*args: ArgsT.args, **kwargs: ArgsT.kwargs) -> RetT:
                locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = enabled
                return fn(*args, **kwargs)
            return wrapper
    return decorator