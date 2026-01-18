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
def _div_param_param(a, b):
    if a.is_constant():
        a = a.value
        if b.is_constant():
            return a / b.value
        if not a and a in _zero_one_optimizations:
            return 0
    elif b.is_constant():
        b = b.value
        if b in _zero_one_optimizations and b:
            return a
        if not b:
            raise ZeroDivisionError()
    return NPV_DivisionExpression((a, b))