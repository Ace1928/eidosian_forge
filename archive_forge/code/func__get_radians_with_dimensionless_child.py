import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _get_radians_with_dimensionless_child(self, node, child_units):
    """
        Return (and test) the units corresponding to an inverse trig expression node
        in the expression tree. Checks that the length of child_units is 1
        and that the child argument is dimensionless, and returns radians

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
    assert len(child_units) == 1
    if self._equivalent_to_dimensionless(child_units[0]):
        return self._pyomo_units_container._pint_registry.radian
    raise UnitsError(f'Expected dimensionless argument to function in expression {node}, but found {child_units[0]}')