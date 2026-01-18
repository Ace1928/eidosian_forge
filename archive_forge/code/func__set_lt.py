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
def _set_lt(typingctx, a, b):
    sig = types.boolean(a, b)

    def codegen(context, builder, sig, args):
        inst = SetInstance(context, builder, sig.args[0], args[0])
        other = SetInstance(context, builder, sig.args[1], args[1])
        return inst.issubset(other, strict=True)
    return (sig, codegen)