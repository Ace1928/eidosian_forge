import math
import random
import numpy as np
from llvmlite import ir
from numba.core.cgutils import is_nonelike
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core.imputils import (Registry, impl_ret_untracked,
from numba.core.typing import signature
from numba.core import types, cgutils
from numba.np import arrayobj
from numba.core.errors import NumbaTypeError
def _seed_impl(state_type):

    @intrinsic
    def _impl(typingcontext, seed):

        def codegen(context, builder, sig, args):
            seed_value, = args
            fnty = ir.FunctionType(ir.VoidType(), (rnd_state_ptr_t, int32_t))
            fn = cgutils.get_or_insert_function(builder.function.module, fnty, 'numba_rnd_init')
            builder.call(fn, (get_state_ptr(context, builder, state_type), seed_value))
            return context.get_constant(types.none, None)
        return (signature(types.void, types.uint32), codegen)
    return lambda seed: _impl(seed)