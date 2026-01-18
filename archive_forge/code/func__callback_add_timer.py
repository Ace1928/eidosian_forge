import collections
import functools
import sys
import types as pytypes
import uuid
import weakref
from contextlib import ExitStack
from abc import abstractmethod
from numba import _dispatcher
from numba.core import (
from numba.core.compiler_lock import global_compiler_lock
from numba.core.typeconv.rules import default_type_manager
from numba.core.typing.templates import fold_arguments
from numba.core.typing.typeof import Purpose, typeof
from numba.core.bytecode import get_code_object
from numba.core.caching import NullCache, FunctionCache
from numba.core import entrypoints
from numba.core.retarget import BaseRetarget
import numba.core.event as ev
def _callback_add_timer(self, duration, cres, lock_name):
    md = cres.metadata
    if md is not None:
        timers = md.setdefault('timers', {})
        if lock_name not in timers:
            timers[lock_name] = duration
        else:
            msg = f"'{lock_name} metadata is already defined."
            raise AssertionError(msg)