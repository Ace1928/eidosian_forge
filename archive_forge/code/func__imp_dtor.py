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
def _imp_dtor(self, context, module):
    """Define the dtor for set
        """
    llvoidptr = cgutils.voidptr_t
    llsize_t = context.get_value_type(types.size_t)
    fnty = ir.FunctionType(ir.VoidType(), [llvoidptr, llsize_t, llvoidptr])
    fname = f'.dtor.set.{self._ty.dtype}'
    fn = cgutils.get_or_insert_function(module, fnty, name=fname)
    if fn.is_declaration:
        fn.linkage = 'linkonce_odr'
        builder = ir.IRBuilder(fn.append_basic_block())
        payload = _SetPayload(context, builder, self._ty, fn.args[0])
        with payload._iterate() as loop:
            entry = loop.entry
            context.nrt.decref(builder, self._ty.dtype, entry.key)
        builder.ret_void()
    return fn