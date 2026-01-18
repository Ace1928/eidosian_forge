from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def _prepare_call_to_object_mode(context, builder, pyapi, func, signature, args):
    mod = builder.module
    bb_core_return = builder.append_basic_block('ufunc.core.return')
    ll_int = context.get_value_type(types.int32)
    ll_intp = context.get_value_type(types.intp)
    ll_intp_ptr = ir.PointerType(ll_intp)
    ll_voidptr = context.get_value_type(types.voidptr)
    ll_pyobj = context.get_value_type(types.pyobject)
    fnty = ir.FunctionType(ll_pyobj, [ll_int, ll_intp_ptr, ll_intp_ptr, ll_voidptr, ll_int, ll_int])
    fn_array_new = cgutils.get_or_insert_function(mod, fnty, 'numba_ndarray_new')
    error_pointer = cgutils.alloca_once(builder, ir.IntType(1), name='error')
    builder.store(cgutils.true_bit, error_pointer)
    object_args = []
    object_pointers = []
    for i, (arg, argty) in enumerate(zip(args, signature.args)):
        objptr = cgutils.alloca_once(builder, ll_pyobj, zfill=True)
        object_pointers.append(objptr)
        if isinstance(argty, types.Array):
            arycls = context.make_array(argty)
            array = arycls(context, builder, value=arg)
            zero = Constant(ll_int, 0)
            nd = Constant(ll_int, argty.ndim)
            dims = builder.gep(array._get_ptr_by_name('shape'), [zero, zero])
            strides = builder.gep(array._get_ptr_by_name('strides'), [zero, zero])
            data = builder.bitcast(array.data, ll_voidptr)
            dtype = np.dtype(str(argty.dtype))
            type_num = Constant(ll_int, dtype.num)
            itemsize = Constant(ll_int, dtype.itemsize)
            obj = builder.call(fn_array_new, [nd, dims, strides, data, type_num, itemsize])
        else:
            obj = pyapi.from_native_value(argty, arg)
        builder.store(obj, objptr)
        object_args.append(obj)
        obj_is_null = cgutils.is_null(builder, obj)
        builder.store(obj_is_null, error_pointer)
        cgutils.cbranch_or_continue(builder, obj_is_null, bb_core_return)
    object_sig = [types.pyobject] * len(object_args)
    status, retval = context.call_conv.call_function(builder, func, types.pyobject, object_sig, object_args)
    builder.store(status.is_error, error_pointer)
    pyapi.decref(retval)
    builder.branch(bb_core_return)
    builder.position_at_end(bb_core_return)
    for objptr in object_pointers:
        pyapi.decref(builder.load(objptr))
    innercall = status.code
    return (innercall, builder.load(error_pointer))