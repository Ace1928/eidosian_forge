import operator
from numba.core import types
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function_template
@register
class Cuda_const_array_like(CallableTemplate):
    key = cuda.const.array_like

    def generic(self):

        def typer(ndarray):
            return ndarray
        return typer