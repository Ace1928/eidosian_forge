import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
@intrinsic
def _set_discard(typingctx, s, item):
    sig = types.none(s, item)

    def set_discard(context, builder, sig, args):
        inst = SetInstance(context, builder, sig.args[0], args[0])
        item = args[1]
        inst.discard(item)
        return context.get_dummy_value()
    return (sig, set_discard)