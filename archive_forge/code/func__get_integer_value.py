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
def _get_integer_value(n, node):
    if n.__class__ in EXPR.native_numeric_types and int(n) == n:
        return n
    if n.__class__ not in EXPR.native_types:
        if n.is_potentially_variable():
            raise MouseTrap("The first argument '%s' to '%s' is potentially variable. This may be a mathematically coherent expression; However it is not yet supported to convert it to a disjunctive program." % (n, node))
        else:
            return n
    raise ValueError("The first argument to '%s' must be an integer.\n\tRecieved: %s" % (node, n))