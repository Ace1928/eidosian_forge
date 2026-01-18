import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _get_unit_for_unary_function(self, node, child_units):
    """
        Return (and test) the units corresponding to a unary function expression node
        in the expression tree. Checks that child_units is of length 1
        and calls the appropriate method from the unary function method map.

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
    func_name = node.getname()
    node_func = self.unary_function_method_map.get(func_name, None)
    if node_func is None:
        raise TypeError(f'An unhandled unary function: {func_name} was encountered while retrieving the units of expression {node}')
    return node_func(self, node, child_units)