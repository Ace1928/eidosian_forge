import operator
from numba.core import types
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function_template
@register_attr
class CudaModuleTemplate(AttributeTemplate):
    key = types.Module(cuda)

    def resolve_cg(self, mod):
        return types.Module(cuda.cg)

    def resolve_threadIdx(self, mod):
        return dim3

    def resolve_blockIdx(self, mod):
        return dim3

    def resolve_blockDim(self, mod):
        return dim3

    def resolve_gridDim(self, mod):
        return dim3

    def resolve_laneid(self, mod):
        return types.int32

    def resolve_shared(self, mod):
        return types.Module(cuda.shared)

    def resolve_popc(self, mod):
        return types.Function(Cuda_popc)

    def resolve_brev(self, mod):
        return types.Function(Cuda_brev)

    def resolve_clz(self, mod):
        return types.Function(Cuda_clz)

    def resolve_ffs(self, mod):
        return types.Function(Cuda_ffs)

    def resolve_fma(self, mod):
        return types.Function(Cuda_fma)

    def resolve_cbrt(self, mod):
        return types.Function(Cuda_cbrt)

    def resolve_threadfence(self, mod):
        return types.Function(Cuda_threadfence_device)

    def resolve_threadfence_block(self, mod):
        return types.Function(Cuda_threadfence_block)

    def resolve_threadfence_system(self, mod):
        return types.Function(Cuda_threadfence_system)

    def resolve_syncwarp(self, mod):
        return types.Function(Cuda_syncwarp)

    def resolve_shfl_sync_intrinsic(self, mod):
        return types.Function(Cuda_shfl_sync_intrinsic)

    def resolve_vote_sync_intrinsic(self, mod):
        return types.Function(Cuda_vote_sync_intrinsic)

    def resolve_match_any_sync(self, mod):
        return types.Function(Cuda_match_any_sync)

    def resolve_match_all_sync(self, mod):
        return types.Function(Cuda_match_all_sync)

    def resolve_activemask(self, mod):
        return types.Function(Cuda_activemask)

    def resolve_lanemask_lt(self, mod):
        return types.Function(Cuda_lanemask_lt)

    def resolve_selp(self, mod):
        return types.Function(Cuda_selp)

    def resolve_nanosleep(self, mod):
        return types.Function(Cuda_nanosleep)

    def resolve_atomic(self, mod):
        return types.Module(cuda.atomic)

    def resolve_fp16(self, mod):
        return types.Module(cuda.fp16)

    def resolve_const(self, mod):
        return types.Module(cuda.const)

    def resolve_local(self, mod):
        return types.Module(cuda.local)