from numba.core import types
from numba.core.extending import overload, overload_method
from numba.core.typing import signature
from numba.cuda import nvvmutils
from numba.cuda.extending import intrinsic
from numba.cuda.types import grid_group, GridGroup as GridGroupClass
@intrinsic
def _grid_group_sync(typingctx, group):
    sig = signature(types.int32, group)

    def codegen(context, builder, sig, args):
        flags = context.get_constant(types.int32, 0)
        mod = builder.module
        return builder.call(nvvmutils.declare_cudaCGSynchronize(mod), (*args, flags))
    return (sig, codegen)