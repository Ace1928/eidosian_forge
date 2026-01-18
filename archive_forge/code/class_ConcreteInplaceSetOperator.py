import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@infer_global(op_key)
class ConcreteInplaceSetOperator(SetOperator):
    key = op_key