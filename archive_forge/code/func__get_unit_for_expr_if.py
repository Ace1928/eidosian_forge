import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _get_unit_for_expr_if(self, node, child_units):
    """
        Return (and test) the units corresponding to an Expr_if expression node
        in the expression tree. The _if relational expression is validated and
        the _then/_else are checked to ensure they have the same units. Also checks
        to make sure length of child_units is 3

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
    assert len(child_units) == 3
    if not self._equivalent_pint_units(child_units[1], child_units[2]):
        raise InconsistentUnitsError(child_units[1], child_units[2], 'Error in units found in expression: %s' % (node,))
    return child_units[1]