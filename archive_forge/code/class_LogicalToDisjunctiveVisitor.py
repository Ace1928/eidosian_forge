import collections
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import MouseTrap
from pyomo.core.expr.expr_common import ExpressionType
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numeric_expr import NumericExpression
from pyomo.core.expr.relational_expr import RelationalExpression
import pyomo.core.expr as EXPR
from pyomo.core.base import (
import pyomo.core.base.boolean_var as BV
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.param import ScalarParam, _ParamData
from pyomo.core.base.var import ScalarVar, _GeneralVarData
from pyomo.gdp.disjunct import AutoLinkedBooleanVar, Disjunct, Disjunction
class LogicalToDisjunctiveVisitor(StreamBasedExpressionVisitor):
    """Converts BooleanExpressions to Linear (MIP) representation

    This converter eschews conjunctive normal form, and instead follows
    the well-trodden MINLP path of factorable programming.

    """

    def __init__(self):
        super().__init__()
        self.z_vars = VarList(domain=Binary)
        self.z_vars.construct()
        self.constraints = ConstraintList()
        self.disjuncts = Disjunct(NonNegativeIntegers, concrete=True)
        self.disjunctions = Disjunction(NonNegativeIntegers)
        self.disjunctions.construct()
        self.expansions = ComponentMap()
        self.boolean_to_binary_map = ComponentMap()

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return (False, self.finalizeResult(result))
        return (True, expr)

    def beforeChild(self, node, child, child_idx):
        if child.__class__ in EXPR.native_types:
            return (False, child)
        if child.is_numeric_type():
            return (False, child)
        if child.is_expression_type(ExpressionType.RELATIONAL):
            return _before_relational_expr(self, child)
        if not child.is_expression_type() or child.is_named_expression_type():
            return _before_child_dispatcher[child.__class__](self, child)
        return (True, None)

    def exitNode(self, node, data):
        return _operator_dispatcher[node.__class__](self, node, *data)

    def finalizeResult(self, result):
        self.constraints.add(result >= 1)
        return result