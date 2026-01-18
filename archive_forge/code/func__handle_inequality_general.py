import logging
import sys
from operator import itemgetter
from itertools import filterfalse
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.expr import is_fixed, value
from pyomo.core.base.expression import Expression
import pyomo.core.kernel as kernel
from pyomo.repn.util import (
def _handle_inequality_general(visitor, node, arg1, arg2):
    ans = visitor.Result()
    ans.nonlinear = InequalityExpression((to_expression(visitor, arg1), to_expression(visitor, arg2)), node.strict)
    return (_GENERAL, ans)