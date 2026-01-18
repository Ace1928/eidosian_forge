import math
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
@infer_global(math.isfinite)
class Math_isfinite(Math_predicate):
    pass