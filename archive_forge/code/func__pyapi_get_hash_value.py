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
def _pyapi_get_hash_value(self, pyapi, context, builder, item):
    """Python API compatible version of `get_hash_value()`.
        """
    argtypes = [self._ty.dtype]
    resty = types.intp

    def wrapper(val):
        return _get_hash_value_intrinsic(val)
    args = [item]
    sig = typing.signature(resty, *argtypes)
    is_error, retval = pyapi.call_jit_code(wrapper, sig, args)
    with builder.if_then(is_error, likely=False):
        builder.ret(pyapi.get_null_object())
    return retval