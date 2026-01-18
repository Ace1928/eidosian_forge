import math
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
@infer_global(math.copysign)
class Math_copysign(ConcreteTemplate):
    cases = [signature(types.float32, types.float32, types.float32), signature(types.float64, types.float64, types.float64)]