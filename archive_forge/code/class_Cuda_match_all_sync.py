import operator
from numba.core import types
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function_template
@register
class Cuda_match_all_sync(ConcreteTemplate):
    key = cuda.match_all_sync
    cases = [signature(types.Tuple((types.i4, types.b1)), types.i4, types.i4), signature(types.Tuple((types.i4, types.b1)), types.i4, types.i8), signature(types.Tuple((types.i4, types.b1)), types.i4, types.f4), signature(types.Tuple((types.i4, types.b1)), types.i4, types.f8)]