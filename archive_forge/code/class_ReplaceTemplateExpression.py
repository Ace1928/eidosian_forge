import itertools
import logging
import sys
import builtins
from contextlib import nullcontext
from pyomo.common.errors import TemplateExpressionError
from pyomo.core.expr.base import ExpressionBase, ExpressionArgs_Mixin, NPV_Mixin
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.relational_expr import tuple_to_relational_expr
from pyomo.core.expr.visitor import (
class ReplaceTemplateExpression(ExpressionReplacementVisitor):
    template_types = {IndexTemplate, GetItemExpression, Numeric_GetItemExpression, NPV_Numeric_GetItemExpression, Boolean_GetItemExpression, NPV_Boolean_GetItemExpression}

    def __init__(self, substituter, *args, **kwargs):
        kwargs.setdefault('remove_named_expressions', True)
        super().__init__(**kwargs)
        self.substituter = substituter
        self.substituter_args = args

    def beforeChild(self, node, child, child_idx):
        if type(child) in ReplaceTemplateExpression.template_types:
            return (False, self.substituter(child, *self.substituter_args))
        return super().beforeChild(node, child, child_idx)