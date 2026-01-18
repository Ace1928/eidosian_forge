from numba.core import types
from numba.core.extending import overload, overload_method
from numba.core.typing import signature
from numba.cuda import nvvmutils
from numba.cuda.extending import intrinsic
from numba.cuda.types import grid_group, GridGroup as GridGroupClass
@intrinsic
def _this_grid(typingctx):
    sig = signature(grid_group)

    def codegen(context, builder, sig, args):
        one = context.get_constant(types.int32, 1)
        mod = builder.module
        return builder.call(nvvmutils.declare_cudaCGGetIntrinsicHandle(mod), (one,))
    return (sig, codegen)