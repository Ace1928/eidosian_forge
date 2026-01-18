import pyomo.core.expr as EXPR
from pyomo.core import (
from pyomo.core.base.misc import create_name
from pyomo.core.plugins.transform.util import partial
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.util import collectAbstractComponents
import logging
class VarmapVisitor(EXPR.ExpressionReplacementVisitor):

    def __init__(self, varmap):
        super(VarmapVisitor, self).__init__()
        self.varmap = varmap

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return (True, node)
        if node.is_variable_type():
            if node.local_name in self.varmap:
                return (True, self.varmap[node.local_name])
            else:
                return (True, node)
        if isinstance(node, EXPR.LinearExpression):
            with EXPR.nonlinear_expression() as expr:
                for c, v in zip(node.linear_coefs, node.linear_vars):
                    if hasattr(v, 'local_name'):
                        expr += c * self.varmap.get(v.local_name)
                    else:
                        expr += c * v
            return (True, expr)
        return (False, None)