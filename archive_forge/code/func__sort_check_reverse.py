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
def _sort_check_reverse(reverse):
    if isinstance(reverse, types.Omitted):
        rty = reverse.value
    elif isinstance(reverse, types.Optional):
        rty = reverse.type
    else:
        rty = reverse
    if not isinstance(rty, (types.Boolean, types.Integer, int, bool)):
        msg = "an integer is required for 'reverse' (got type %s)" % reverse
        raise errors.TypingError(msg)
    return rty