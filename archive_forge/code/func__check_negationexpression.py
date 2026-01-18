import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def _check_negationexpression(expr, i):
    """
    Accepts an equality expression and an index value. Checks the
    NegationExpression at expr.arg(i) to see if it contains a
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>`. If so, return the
    GetItemExpression for the
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>` and the RHS. If not,
    return None.
    """
    arg = expr.arg(i).arg(0)
    if isinstance(arg, EXPR.GetItemExpression) and type(arg.arg(0)) is DerivativeVar:
        return [arg, -expr.arg(1 - i)]
    if type(arg) is EXPR.ProductExpression:
        lhs = arg.arg(0)
        rhs = arg.arg(1)
        if not (type(lhs) in native_numeric_types or not lhs.is_potentially_variable()):
            return None
        if not (isinstance(rhs, EXPR.GetItemExpression) and type(rhs.arg(0)) is DerivativeVar):
            return None
        return [rhs, -expr.arg(1 - i) / lhs]
    return None