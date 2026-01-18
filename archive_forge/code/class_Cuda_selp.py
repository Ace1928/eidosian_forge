import operator
from numba.core import types
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function_template
@register
class Cuda_selp(AbstractTemplate):
    key = cuda.selp

    def generic(self, args, kws):
        assert not kws
        test, a, b = args
        supported_types = (types.float64, types.float32, types.int16, types.uint16, types.int32, types.uint32, types.int64, types.uint64)
        if a != b or a not in supported_types:
            return
        return signature(a, test, a, a)