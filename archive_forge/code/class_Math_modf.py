import math
from numba.core import types
from numba.core.typing.templates import ConcreteTemplate, signature, Registry
@infer_global(math.modf)
class Math_modf(ConcreteTemplate):
    cases = [signature(types.UniTuple(types.float64, 2), types.float64), signature(types.UniTuple(types.float32, 2), types.float32)]