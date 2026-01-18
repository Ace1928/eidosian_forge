import math
from numba.core import types
from numba.core.typing.templates import ConcreteTemplate, signature, Registry
@infer_global(math.remainder)
class Math_remainder(ConcreteTemplate):
    cases = [signature(types.float32, types.float32, types.float32), signature(types.float64, types.float64, types.float64)]