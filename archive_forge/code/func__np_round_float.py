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
def _np_round_float(typingctx, val):
    sig = val(val)

    def codegen(context, builder, sig, args):
        [val] = args
        tp = sig.args[0]
        llty = context.get_value_type(tp)
        module = builder.module
        fnty = llvmlite.ir.FunctionType(llty, [llty])
        fn = cgutils.get_or_insert_function(module, fnty, _np_round_intrinsic(tp))
        res = builder.call(fn, (val,))
        return impl_ret_untracked(context, builder, sig.return_type, res)
    return (sig, codegen)