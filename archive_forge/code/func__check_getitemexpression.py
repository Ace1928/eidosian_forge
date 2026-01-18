import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def _check_getitemexpression(expr, i):
    """
    Accepts an equality expression and an index value. Checks the
    GetItemExpression at expr.arg(i) to see if it is a
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>`. If so, return the
    GetItemExpression for the :py:class:`DerivativeVar<DerivativeVar>` and
    the RHS. If not, return None.
    """
    if type(expr.arg(i).arg(0)) is DerivativeVar:
        return [expr.arg(i), expr.arg(1 - i)]
    else:
        return None