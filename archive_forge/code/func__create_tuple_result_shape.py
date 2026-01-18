import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
@intrinsic
def _create_tuple_result_shape(tyctx, shape_list, shape_tuple):
    """
    This routine converts shape list where the axis dimension has already
    been popped to a tuple for indexing of the same size.  The original shape
    tuple is also required because it contains a length field at compile time
    whereas the shape list does not.
    """
    nd = len(shape_tuple) - 1
    tupty = types.UniTuple(types.intp, nd)
    function_sig = tupty(shape_list, shape_tuple)

    def codegen(cgctx, builder, signature, args):
        lltupty = cgctx.get_value_type(tupty)
        tup = cgutils.get_null_value(lltupty)
        [in_shape, _] = args

        def array_indexer(a, i):
            return a[i]
        for i in range(nd):
            dataidx = cgctx.get_constant(types.intp, i)
            data = cgctx.compile_internal(builder, array_indexer, types.intp(shape_list, types.intp), [in_shape, dataidx])
            tup = builder.insert_value(tup, data, i)
        return tup
    return (function_sig, codegen)