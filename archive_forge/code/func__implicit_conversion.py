import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
def _implicit_conversion(obj):
    if isinstance(obj, (int, float)):
        return Constant(obj)
    elif isinstance(obj, Expr):
        return obj
    elif isinstance(obj, str):
        return Symbol(unique_keys=(obj,))
    if sympy is not None:
        if isinstance(obj, sympy.Mul):
            if len(obj.args) != 2:
                raise NotImplementedError('Did you use evaluate=False?')
            return _MulExpr([_implicit_conversion(obj.args[0]), _implicit_conversion(obj.args[1])])
        elif isinstance(obj, sympy.Add):
            if len(obj.args) != 2:
                raise NotImplementedError('Did you use evaluate=False?')
            return _AddExpr([_implicit_conversion(obj.args[0]), _implicit_conversion(obj.args[1])])
        elif isinstance(obj, sympy.Pow):
            return _PowExpr(_implicit_conversion(obj.base), _implicit_conversion(obj.exp))
        elif isinstance(obj, sympy.Float):
            return Constant(float(obj))
        elif isinstance(obj, sympy.Symbol):
            return Symbol(unique_keys=(obj.name,))
    raise NotImplementedError("Don't know how to convert %s (of type %s)" % (obj, type(obj)))