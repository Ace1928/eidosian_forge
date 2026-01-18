import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _get_unit_for_pow(self, node, child_units):
    """
        Return (and test) the units corresponding to a pow expression node
        in the expression tree.

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
    assert len(child_units) == 2
    if not self._equivalent_to_dimensionless(child_units[1]):
        raise UnitsError(f'Error in sub-expression: {node}. Exponents in a pow expression must be dimensionless.')
    exponent = node.args[1]
    if type(exponent) in nonpyomo_leaf_types:
        return child_units[0] ** value(exponent)
    if self._equivalent_to_dimensionless(child_units[0]):
        return self._pint_dimensionless
    if not exponent.is_fixed():
        raise UnitsError(f'The base of an exponent has units {child_units[0]}, but the exponent is not a fixed numerical value.')
    return child_units[0] ** value(exponent)