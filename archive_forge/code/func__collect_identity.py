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
def _collect_identity(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if exp._args_[0].__class__ in native_numeric_types:
        return Results(constant=multiplier * exp._args_[0])
    if not exp._args_[0].is_potentially_variable():
        if compute_values:
            return Results(constant=multiplier * value(exp._args_[0]))
        else:
            return Results(constant=multiplier * exp._args_[0])
    return _collect_standard_repn(exp.expr, multiplier, idMap, compute_values, verbose, quadratic)