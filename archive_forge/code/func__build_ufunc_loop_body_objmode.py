from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def _build_ufunc_loop_body_objmode(load, store, context, func, builder, arrays, out, offsets, store_offset, signature, env, pyapi):
    elems = load()
    _objargs = [types.pyobject] * len(signature.args)
    with pyapi.err_push(keep_new=True):
        status, retval = context.call_conv.call_function(builder, func, types.pyobject, _objargs, elems)
        for elem in elems:
            pyapi.decref(elem)
    store(retval)
    for off, ary in zip(offsets, arrays):
        builder.store(builder.add(builder.load(off), ary.step), off)
    builder.store(builder.add(builder.load(store_offset), out.step), store_offset)
    return status.code