import math
from numba.core import types
from numba.core.typing.templates import ConcreteTemplate, signature, Registry
@infer_global(math.isinf)
@infer_global(math.isnan)
@infer_global(math.isfinite)
class Math_isnan(ConcreteTemplate):
    cases = [signature(types.boolean, types.int64), signature(types.boolean, types.uint64), signature(types.boolean, types.float32), signature(types.boolean, types.float64)]