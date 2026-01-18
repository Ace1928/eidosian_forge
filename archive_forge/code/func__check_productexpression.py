import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def _check_productexpression(expr, i):
    """
    Accepts an equality expression and an index value. Checks the
    ProductExpression at expr.arg(i) to see if it contains a
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>`. If so, return the
    GetItemExpression for the
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>` and the RHS. If not,
    return None.
    """
    expr_ = expr.arg(i)
    stack = [(expr_, 1)]
    pterms = []
    dv = None
    while stack:
        curr, e_ = stack.pop()
        if curr.__class__ is EXPR.ProductExpression:
            stack.append((curr.arg(0), e_))
            stack.append((curr.arg(1), e_))
        elif curr.__class__ is EXPR.DivisionExpression:
            stack.append((curr.arg(0), e_))
            stack.append((curr.arg(1), -e_))
        elif isinstance(curr, EXPR.GetItemExpression) and type(curr.arg(0)) is DerivativeVar:
            dv = (curr, e_)
        else:
            pterms.append((curr, e_))
    if dv is None:
        return None
    numerator = 1
    denom = 1
    for term, e_ in pterms:
        if e_ == 1:
            denom *= term
        else:
            numerator *= term
    curr, e_ = dv
    if e_ == 1:
        return [curr, expr.arg(1 - i) * numerator / denom]
    else:
        return [curr, denom / (expr.arg(1 - i) * numerator)]