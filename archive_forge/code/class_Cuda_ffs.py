import operator
from numba.core import types
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function_template
@register
class Cuda_ffs(ConcreteTemplate):
    """
    Supported types from `llvm.cttz`
    [here](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics)
    """
    key = cuda.ffs
    cases = [signature(types.uint32, types.int8), signature(types.uint32, types.int16), signature(types.uint32, types.int32), signature(types.uint32, types.int64), signature(types.uint32, types.uint8), signature(types.uint32, types.uint16), signature(types.uint32, types.uint32), signature(types.uint32, types.uint64)]