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
def _collect_branching_expr(exp, multiplier, idMap, compute_values, verbose, quadratic):
    _if, _then, _else = exp.args
    if _if.__class__ in native_types:
        if_val = _if
    elif not _if.is_potentially_variable():
        if compute_values:
            if_val = value(_if)
        else:
            return Results(nonl=multiplier * exp)
    else:
        res = _collect_standard_repn(_if, 1, idMap, compute_values, verbose, quadratic)
        if not (res.nonl.__class__ in native_numeric_types and res.nonl == 0) or len(res.linear) > 0 or (quadratic and len(res.quadratic) > 0):
            return Results(nonl=multiplier * exp)
        elif res.constant.__class__ in native_numeric_types:
            if_val = res.constant
        else:
            return Results(constant=multiplier * exp)
    if if_val:
        if _then.__class__ in native_numeric_types:
            return Results(constant=multiplier * _then)
        return _collect_standard_repn(_then, multiplier, idMap, compute_values, verbose, quadratic)
    else:
        if _else.__class__ in native_numeric_types:
            return Results(constant=multiplier * _else)
        return _collect_standard_repn(_else, multiplier, idMap, compute_values, verbose, quadratic)