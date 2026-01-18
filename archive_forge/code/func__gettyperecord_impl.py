from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
@intrinsic
def _gettyperecord_impl(typingctx, codepoint):
    """
    Provides the binding to numba_gettyperecord, returns a `typerecord`
    namedtuple of properties from the codepoint.
    """
    if not isinstance(codepoint, types.Integer):
        raise TypingError('codepoint must be an integer')

    def details(context, builder, signature, args):
        ll_void = context.get_value_type(types.void)
        ll_Py_UCS4 = context.get_value_type(_Py_UCS4)
        ll_intc = context.get_value_type(types.intc)
        ll_intc_ptr = ll_intc.as_pointer()
        ll_uchar = context.get_value_type(types.uchar)
        ll_uchar_ptr = ll_uchar.as_pointer()
        ll_ushort = context.get_value_type(types.ushort)
        ll_ushort_ptr = ll_ushort.as_pointer()
        fnty = llvmlite.ir.FunctionType(ll_void, [ll_Py_UCS4, ll_intc_ptr, ll_intc_ptr, ll_intc_ptr, ll_uchar_ptr, ll_uchar_ptr, ll_ushort_ptr])
        fn = cgutils.get_or_insert_function(builder.module, fnty, name='numba_gettyperecord')
        upper = cgutils.alloca_once(builder, ll_intc, name='upper')
        lower = cgutils.alloca_once(builder, ll_intc, name='lower')
        title = cgutils.alloca_once(builder, ll_intc, name='title')
        decimal = cgutils.alloca_once(builder, ll_uchar, name='decimal')
        digit = cgutils.alloca_once(builder, ll_uchar, name='digit')
        flags = cgutils.alloca_once(builder, ll_ushort, name='flags')
        byref = [upper, lower, title, decimal, digit, flags]
        builder.call(fn, [args[0]] + byref)
        buf = []
        for x in byref:
            buf.append(builder.load(x))
        res = context.make_tuple(builder, signature.return_type, tuple(buf))
        return impl_ret_untracked(context, builder, signature.return_type, res)
    tupty = types.NamedTuple([types.intc, types.intc, types.intc, types.uchar, types.uchar, types.ushort], typerecord)
    sig = tupty(_Py_UCS4)
    return (sig, details)