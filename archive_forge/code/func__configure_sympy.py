import operator
import sys
from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import NondifferentiableError
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import value, native_types
def _configure_sympy(sympy, available):
    if not available:
        return
    _operatorMap.update({sympy.Add: sum, sympy.Mul: _prod, sympy.Pow: lambda x: operator.pow(*x), sympy.exp: lambda x: EXPR.exp(*x), sympy.log: lambda x: EXPR.log(*x), sympy.sin: lambda x: EXPR.sin(*x), sympy.asin: lambda x: EXPR.asin(*x), sympy.sinh: lambda x: EXPR.sinh(*x), sympy.asinh: lambda x: EXPR.asinh(*x), sympy.cos: lambda x: EXPR.cos(*x), sympy.acos: lambda x: EXPR.acos(*x), sympy.cosh: lambda x: EXPR.cosh(*x), sympy.acosh: lambda x: EXPR.acosh(*x), sympy.tan: lambda x: EXPR.tan(*x), sympy.atan: lambda x: EXPR.atan(*x), sympy.tanh: lambda x: EXPR.tanh(*x), sympy.atanh: lambda x: EXPR.atanh(*x), sympy.ceiling: lambda x: EXPR.ceil(*x), sympy.floor: lambda x: EXPR.floor(*x), sympy.sqrt: lambda x: EXPR.sqrt(*x), sympy.Abs: lambda x: abs(*x), sympy.Derivative: _nondifferentiable, sympy.Tuple: lambda x: x, sympy.Or: lambda x: EXPR.lor(*x), sympy.And: lambda x: EXPR.land(*x), sympy.Implies: lambda x: EXPR.implies(*x), sympy.Equivalent: lambda x: EXPR.equivalents(*x), sympy.Not: lambda x: EXPR.lnot(*x), sympy.LessThan: lambda x: operator.le(*x), sympy.StrictLessThan: lambda x: operator.lt(*x), sympy.GreaterThan: lambda x: operator.ge(*x), sympy.StrictGreaterThan: lambda x: operator.gt(*x), sympy.Equality: lambda x: operator.eq(*x)})
    _pyomo_operator_map.update({EXPR.SumExpression: sympy.Add, EXPR.LinearExpression: sympy.Add, EXPR.ProductExpression: sympy.Mul, EXPR.MonomialTermExpression: sympy.Mul, EXPR.ExternalFunctionExpression: _external_fcn, EXPR.AndExpression: sympy.And, EXPR.OrExpression: sympy.Or, EXPR.ImplicationExpression: sympy.Implies, EXPR.EquivalenceExpression: sympy.Equivalent, EXPR.XorExpression: sympy.Xor, EXPR.NotExpression: sympy.Not})
    _functionMap.update({'exp': sympy.exp, 'log': sympy.log, 'log10': lambda x: sympy.log(x) / sympy.log(10), 'sin': sympy.sin, 'asin': sympy.asin, 'sinh': sympy.sinh, 'asinh': sympy.asinh, 'cos': sympy.cos, 'acos': sympy.acos, 'cosh': sympy.cosh, 'acosh': sympy.acosh, 'tan': sympy.tan, 'atan': sympy.atan, 'tanh': sympy.tanh, 'atanh': sympy.atanh, 'ceil': sympy.ceiling, 'floor': sympy.floor, 'sqrt': sympy.sqrt})