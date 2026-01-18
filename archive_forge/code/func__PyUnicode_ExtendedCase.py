from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
@intrinsic
def _PyUnicode_ExtendedCase(typingctx, index):
    """
    Accessor function for the _PyUnicode_ExtendedCase array, binds to
    numba_get_PyUnicode_ExtendedCase which wraps the array and does the lookup
    """
    if not isinstance(index, types.Integer):
        raise TypingError('Expected an index')

    def details(context, builder, signature, args):
        ll_Py_UCS4 = context.get_value_type(_Py_UCS4)
        ll_intc = context.get_value_type(types.intc)
        fnty = llvmlite.ir.FunctionType(ll_Py_UCS4, [ll_intc])
        fn = cgutils.get_or_insert_function(builder.module, fnty, name='numba_get_PyUnicode_ExtendedCase')
        return builder.call(fn, [args[0]])
    sig = _Py_UCS4(types.intc)
    return (sig, details)