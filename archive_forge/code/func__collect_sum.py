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
def _collect_sum(exp, multiplier, idMap, compute_values, verbose, quadratic):
    ans = Results()
    nonl = []
    varkeys = idMap[None]
    for e_ in itertools.islice(exp._args_, exp.nargs()):
        if e_.__class__ is EXPR.MonomialTermExpression:
            lhs, v = e_.args
            if lhs.__class__ in native_numeric_types:
                if not lhs:
                    continue
            elif compute_values:
                lhs = value(lhs)
                if not lhs:
                    continue
            if v.fixed:
                if compute_values:
                    ans.constant += multiplier * lhs * value(v)
                else:
                    ans.constant += multiplier * lhs * v
            else:
                id_ = id(v)
                if id_ in varkeys:
                    key = varkeys[id_]
                else:
                    key = len(idMap) - 1
                    varkeys[id_] = key
                    idMap[key] = v
                if key in ans.linear:
                    ans.linear[key] += multiplier * lhs
                else:
                    ans.linear[key] = multiplier * lhs
        elif e_.__class__ in native_numeric_types:
            ans.constant += multiplier * e_
        elif e_.is_variable_type():
            if e_.fixed:
                if compute_values:
                    ans.constant += multiplier * e_.value
                else:
                    ans.constant += multiplier * e_
            else:
                id_ = id(e_)
                if id_ in varkeys:
                    key = varkeys[id_]
                else:
                    key = len(idMap) - 1
                    varkeys[id_] = key
                    idMap[key] = e_
                if key in ans.linear:
                    ans.linear[key] += multiplier
                else:
                    ans.linear[key] = multiplier
        elif not e_.is_potentially_variable():
            if compute_values:
                ans.constant += multiplier * value(e_)
            else:
                ans.constant += multiplier * e_
        else:
            res_ = _collect_standard_repn(e_, multiplier, idMap, compute_values, verbose, quadratic)
            ans.constant += res_.constant
            if not (res_.nonl.__class__ in native_numeric_types and res_.nonl == 0):
                nonl.append(res_.nonl)
            for i, v in res_.linear.items():
                ans.linear[i] = ans.linear.get(i, 0) + v
            if quadratic:
                for i in res_.quadratic:
                    ans.quadratic[i] = ans.quadratic.get(i, 0) + res_.quadratic[i]
    if len(nonl) > 0:
        if len(nonl) == 1:
            ans.nonl = nonl[0]
        else:
            ans.nonl = EXPR.SumExpression(nonl)
    zero_coef = [k for k, coef in ans.linear.items() if coef.__class__ in native_numeric_types and (not coef)]
    for k in zero_coef:
        ans.linear.pop(k)
    return ans