import math
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
@infer_global(math.trunc)
class Math_converter(ConcreteTemplate):
    cases = [signature(types.intp, types.intp), signature(types.int64, types.int64), signature(types.uint64, types.uint64), signature(types.int64, types.float32), signature(types.int64, types.float64)]