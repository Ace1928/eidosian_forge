import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _get_dimensionless_with_dimensionless_children(self, node, child_units):
    """
        Check to make sure that any child arguments are unitless /
        dimensionless (for functions like exp()) and return dimensionless.

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
    for pint_unit in child_units:
        if not self._equivalent_to_dimensionless(pint_unit):
            raise UnitsError(f'Expected no units or dimensionless units in {node}, but found {pint_unit}.')
    return self._pint_dimensionless