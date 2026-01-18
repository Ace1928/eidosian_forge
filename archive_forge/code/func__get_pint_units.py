import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _get_pint_units(self, expr):
    """
        Return the pint units corresponding to the expression. This does
        a number of checks as well.

        Parameters
        ----------
        expr : Pyomo expression
           the input expression for extracting units

        Returns
        -------
        : pint unit
        """
    if expr is None:
        return self._pint_dimensionless
    return self._pintUnitExtractionVisitor.walk_expression(expr=expr)