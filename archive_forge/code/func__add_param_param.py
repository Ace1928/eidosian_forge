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
def _add_param_param(a, b):
    if a.is_constant():
        a = a.value
        if b.is_constant():
            return a + b.value
        elif not a:
            return b
    elif b.is_constant():
        b = b.value
        if not b:
            return a
    return NPV_SumExpression([a, b])