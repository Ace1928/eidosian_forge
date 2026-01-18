import operator
from pyomo.common.deprecation import deprecated
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.numeric_types import (
from .base import ExpressionBase
from .boolean_value import BooleanValue
from .expr_common import _lt, _le, _eq, ExpressionType
from .numvalue import is_potentially_variable, is_constant
from .visitor import polynomial_degree
def _process_relational_arg(arg, n):
    try:
        _numeric = arg.is_numeric_type()
    except AttributeError:
        _numeric = False
    if _numeric:
        if arg.is_constant():
            arg = value(arg)
        else:
            _process_relational_arg.constant = False
    elif arg.__class__ is InequalityExpression:
        _process_relational_arg.relational += n
        _process_relational_arg.constant = False
    else:
        arg = _process_nonnumeric_arg(arg)
        if arg.__class__ not in native_numeric_types:
            _process_relational_arg.constant = False
    return arg