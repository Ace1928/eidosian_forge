import sys
import logging
import itertools
from pyomo.common.numeric_types import native_types, native_numeric_types
from pyomo.core.base import Constraint, Objective, ComponentMap
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.base.objective import _GeneralObjectiveData, ScalarObjective
from pyomo.core.base import _ExpressionData, Expression
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.var import ScalarVar, Var, _GeneralVarData, value
from pyomo.core.base.param import ScalarParam, _ParamData
from pyomo.core.kernel.expression import expression, noclone
from pyomo.core.kernel.variable import IVariable, variable
from pyomo.core.kernel.objective import objective
from io import StringIO
def _collect_nonl(exp, multiplier, idMap, compute_values, verbose, quadratic):
    res = _collect_standard_repn(exp._args_[0], 1, idMap, compute_values, verbose, quadratic)
    if not (res.nonl.__class__ in native_numeric_types and res.nonl == 0) or len(res.linear) > 0 or (quadratic and len(res.quadratic) > 0):
        return Results(nonl=multiplier * exp)
    if compute_values:
        return Results(constant=multiplier * exp._apply_operation([res.constant]))
    else:
        return Results(constant=multiplier * exp)