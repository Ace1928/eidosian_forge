import math
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
@infer_global(math.floor)
@infer_global(math.ceil)
class Math_floor_ceil(Math_converter):
    pass