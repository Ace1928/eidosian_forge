import os
import platform
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import (
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual
class ReplacementWalkerTest3(ExpressionReplacementVisitor):

    def __init__(self, model):
        super().__init__(remove_named_expressions=False)
        self.model = model

    def visiting_potential_leaf(self, node):
        if node.__class__ in (_ParamData, ScalarParam):
            if id(node) in self.substitute:
                return (True, self.substitute[id(node)])
            self.substitute[id(node)] = 2 * self.model.w.add()
            return (True, self.substitute[id(node)])
        if node.__class__ in nonpyomo_leaf_types or node.is_constant() or node.is_variable_type():
            return (True, node)
        return (False, None)