import operator
from numba.core import types
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function_template
@register
class Cuda_atomic_cas(AbstractTemplate):
    key = cuda.atomic.cas

    def generic(self, args, kws):
        assert not kws
        ary, idx, old, val = args
        dty = ary.dtype
        if dty not in integer_numba_types:
            return
        if ary.ndim == 1:
            return signature(dty, ary, types.intp, dty, dty)
        elif ary.ndim > 1:
            return signature(dty, ary, idx, dty, dty)