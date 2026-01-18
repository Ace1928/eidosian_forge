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
def _collect_division(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if exp._args_[1].__class__ in native_numeric_types or not exp._args_[1].is_potentially_variable():
        if compute_values:
            denom = 1.0 * value(exp._args_[1])
        else:
            denom = 1.0 * exp._args_[1]
    else:
        res = _collect_standard_repn(exp._args_[1], 1, idMap, compute_values, verbose, quadratic)
        if not (res.nonl.__class__ in native_numeric_types and res.nonl == 0) or len(res.linear) > 0 or (quadratic and len(res.quadratic) > 0):
            return Results(nonl=multiplier * exp)
        else:
            denom = 1.0 * res.constant
    if denom.__class__ in native_numeric_types and denom == 0:
        raise ZeroDivisionError
    if exp._args_[0].__class__ in native_numeric_types or not exp._args_[0].is_potentially_variable():
        num = exp._args_[0]
        if compute_values:
            num = value(num)
        return Results(constant=multiplier * num / denom)
    return _collect_standard_repn(exp._args_[0], multiplier / denom, idMap, compute_values, verbose, quadratic)