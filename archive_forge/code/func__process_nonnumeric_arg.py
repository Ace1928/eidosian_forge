import operator
from pyomo.common.deprecation import deprecated
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.numeric_types import (
from .base import ExpressionBase
from .boolean_value import BooleanValue
from .expr_common import _lt, _le, _eq, ExpressionType
from .numvalue import is_potentially_variable, is_constant
from .visitor import polynomial_degree
def _process_nonnumeric_arg(obj):
    if hasattr(obj, 'as_numeric'):
        obj = obj.as_numeric()
    elif check_if_numeric_type(obj):
        return obj
    else:
        if obj.is_component_type() and obj.is_indexed():
            raise TypeError('Argument for expression is an indexed numeric value\nspecified without an index:\n\t%s\nIs this value defined over an index that you did not specify?' % (obj.name,))
        raise TypeError('Attempting to use a non-numeric type (%s) in a numeric expression context.' % (obj.__class__.__name__,))