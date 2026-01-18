import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
class Pyomo2Scipy_Visitor(EXPR.ExpressionReplacementVisitor):
    """
    Expression walker that replaces _GetItemExpression
    instances with mutable parameters.
    """

    def __init__(self, templatemap):
        super().__init__(descend_into_named_expressions=True, remove_named_expressions=True)
        self.templatemap = templatemap

    def beforeChild(self, node, child, child_idx):
        if type(child) is IndexTemplate:
            return (False, child)
        if isinstance(child, EXPR.GetItemExpression):
            _id = _GetItemIndexer(child)
            if _id not in self.templatemap:
                self.templatemap[_id] = Param(mutable=True)
                self.templatemap[_id].construct()
                self.templatemap[_id]._name = '%s[%s]' % (_id.base.name, ','.join((str(x) for x in _id.args)))
            return (False, self.templatemap[_id])
        return super().beforeChild(node, child, child_idx)