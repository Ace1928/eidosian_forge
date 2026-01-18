import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _get_dimensionless_no_children(self, node, child_units):
    """
        Check to make sure the length of child_units is zero, and returns
        dimensionless. Used for leaf nodes that should not have any units.

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
    assert len(child_units) == 0
    assert type(node) is IndexTemplate
    return self._pint_dimensionless