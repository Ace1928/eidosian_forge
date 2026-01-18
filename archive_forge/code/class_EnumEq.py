import operator
from numba.core import types
from numba.core.typing.templates import (AbstractTemplate, AttributeTemplate,
@infer_global(operator.eq)
class EnumEq(EnumCompare):
    pass