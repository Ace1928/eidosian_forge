import operator
from numba.core import types
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function_template
@register_attr
class CudaFp16Template(AttributeTemplate):
    key = types.Module(cuda.fp16)

    def resolve_hadd(self, mod):
        return types.Function(Cuda_hadd)

    def resolve_hsub(self, mod):
        return types.Function(Cuda_hsub)

    def resolve_hmul(self, mod):
        return types.Function(Cuda_hmul)

    def resolve_hdiv(self, mod):
        return hdiv_device

    def resolve_hneg(self, mod):
        return types.Function(Cuda_hneg)

    def resolve_habs(self, mod):
        return types.Function(Cuda_habs)

    def resolve_hfma(self, mod):
        return types.Function(Cuda_hfma)

    def resolve_hsin(self, mod):
        return hsin_device

    def resolve_hcos(self, mod):
        return hcos_device

    def resolve_hlog(self, mod):
        return hlog_device

    def resolve_hlog10(self, mod):
        return hlog10_device

    def resolve_hlog2(self, mod):
        return hlog2_device

    def resolve_hexp(self, mod):
        return hexp_device

    def resolve_hexp10(self, mod):
        return hexp10_device

    def resolve_hexp2(self, mod):
        return hexp2_device

    def resolve_hfloor(self, mod):
        return hfloor_device

    def resolve_hceil(self, mod):
        return hceil_device

    def resolve_hsqrt(self, mod):
        return hsqrt_device

    def resolve_hrsqrt(self, mod):
        return hrsqrt_device

    def resolve_hrcp(self, mod):
        return hrcp_device

    def resolve_hrint(self, mod):
        return hrint_device

    def resolve_htrunc(self, mod):
        return htrunc_device

    def resolve_heq(self, mod):
        return types.Function(Cuda_heq)

    def resolve_hne(self, mod):
        return types.Function(Cuda_hne)

    def resolve_hge(self, mod):
        return types.Function(Cuda_hge)

    def resolve_hgt(self, mod):
        return types.Function(Cuda_hgt)

    def resolve_hle(self, mod):
        return types.Function(Cuda_hle)

    def resolve_hlt(self, mod):
        return types.Function(Cuda_hlt)

    def resolve_hmax(self, mod):
        return types.Function(Cuda_hmax)

    def resolve_hmin(self, mod):
        return types.Function(Cuda_hmin)