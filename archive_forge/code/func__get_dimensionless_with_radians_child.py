import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _get_dimensionless_with_radians_child(self, node, child_units):
    """
        Return (and test) the units corresponding to a trig function expression node
        in the expression tree. Checks that the length of child_units is 1
        and that the units of that child expression are unitless or radians and
        returns dimensionless for the units.

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
        return self._pint_dimensionless
    if self._equivalent_pint_units(child_units[0], self._pyomo_units_container._pint_registry.radian):
        return self._pint_dimensionless
    raise UnitsError('Expected radians or dimensionless in argument to function in expression %s, but found %s' % (node, child_units[0]))