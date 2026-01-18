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
def _legalize_arg_types(self, args):
    for i, a in enumerate(args, start=1):
        if isinstance(a, types.List):
            msg = 'Does not support list type inputs into with-context for arg {}'
            raise errors.TypingError(msg.format(i))
        elif isinstance(a, types.Dispatcher):
            msg = 'Does not support function type inputs into with-context for arg {}'
            raise errors.TypingError(msg.format(i))