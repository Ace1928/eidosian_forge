import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
class Substitute_Pyomo2Casadi_Visitor(EXPR.ExpressionReplacementVisitor):
    """
    Expression walker that replaces

       * _UnaryFunctionExpression instances with unary functions that
         point to casadi intrinsic functions.

       * _GetItemExpressions with _GetItemIndexer objects that references
         CasADi variables.
    """

    def __init__(self, templatemap):
        super().__init__(descend_into_named_expressions=True, remove_named_expressions=True)
        self.templatemap = templatemap

    def exitNode(self, node, data):
        """Replace a node if it's a unary function."""
        ans = super().exitNode(node, data)
        if type(ans) is EXPR.UnaryFunctionExpression:
            return EXPR.UnaryFunctionExpression(ans.args, ans.getname(), casadi_intrinsic[ans.getname()])
        return ans

    def beforeChild(self, node, child, child_idx):
        """Replace a node if it's a _GetItemExpression."""
        if isinstance(child, EXPR.GetItemExpression):
            _id = _GetItemIndexer(child)
            if _id not in self.templatemap:
                name = '%s[%s]' % (_id.base.name, ','.join((str(x) for x in _id.args)))
                self.templatemap[_id] = casadi.SX.sym(name)
            return (False, self.templatemap[_id])
        elif type(child) is IndexTemplate:
            return (False, child)
        return super().beforeChild(node, child, child_idx)