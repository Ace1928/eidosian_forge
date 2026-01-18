import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
def _sort_check_key(key):
    if isinstance(key, types.Optional):
        msg = 'Key must concretely be None or a Numba JIT compiled function, an Optional (union of None and a value) was found'
        raise errors.TypingError(msg)
    if not (cgutils.is_nonelike(key) or isinstance(key, types.Dispatcher)):
        msg = 'Key must be None or a Numba JIT compiled function'
        raise errors.TypingError(msg)