import math
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
@infer_global(math.atan2)
class Math_atan2(ConcreteTemplate):
    cases = [signature(types.float64, types.int64, types.int64), signature(types.float64, types.uint64, types.uint64), signature(types.float32, types.float32, types.float32), signature(types.float64, types.float64, types.float64)]