import math
from pyomo.common.dependencies import attempt_import
from pyomo.core import value, SymbolMap, NumericLabeler, Var, Constraint
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.gdp import Disjunction
def _get_default_functions(self):
    default = list()
    default.append('(define-fun exp ((x Real)) Real (^ %0.15f x))' % (math.exp(1),))
    return default