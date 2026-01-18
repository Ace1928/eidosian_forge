import collections
import enum
import logging
import math
import operator
from pyomo.common.dependencies import attempt_import
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import (
from pyomo.core.expr.base import ExpressionBase, NPV_Mixin, visitor
def _categorize_arg_type(arg):
    if arg.__class__ in _known_arg_types:
        return _known_arg_types[arg.__class__]
    if arg.__class__ in native_numeric_types:
        ans = ARG_TYPE.NATIVE
    else:
        try:
            is_numeric = arg.is_numeric_type()
        except AttributeError:
            if check_if_numeric_type(arg):
                ans = ARG_TYPE.NATIVE
            else:
                ans = ARG_TYPE.INVALID
        else:
            if is_numeric:
                ans = None
            elif hasattr(arg, 'as_numeric'):
                ans = ARG_TYPE.ASNUMERIC
            else:
                ans = ARG_TYPE.INVALID
    if ans is None:
        if arg.is_expression_type():
            if not arg.is_potentially_variable():
                ans = ARG_TYPE.NPV
                NPV_expression_types.add(arg.__class__)
            elif isinstance(arg, _MutableSumExpression):
                ans = ARG_TYPE.MUTABLE
            elif arg.__class__ is MonomialTermExpression:
                ans = ARG_TYPE.MONOMIAL
            elif isinstance(arg, LinearExpression):
                ans = ARG_TYPE.LINEAR
            elif isinstance(arg, SumExpression):
                ans = ARG_TYPE.SUM
            else:
                ans = ARG_TYPE.OTHER
        elif not arg.is_potentially_variable():
            ans = ARG_TYPE.PARAM
        elif arg.is_variable_type():
            ans = ARG_TYPE.VAR
        else:
            ans = ARG_TYPE.OTHER
    register_arg_type(arg.__class__, ans)
    return ans