import operator
from numba.core import types
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function_template
class Cuda_array_decl(CallableTemplate):

    def generic(self):

        def typer(shape, dtype):
            if isinstance(shape, types.Integer):
                if not isinstance(shape, types.IntegerLiteral):
                    return None
            elif isinstance(shape, (types.Tuple, types.UniTuple)):
                if any([not isinstance(s, types.IntegerLiteral) for s in shape]):
                    return None
            else:
                return None
            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                return types.Array(dtype=nb_dtype, ndim=ndim, layout='C')
        return typer