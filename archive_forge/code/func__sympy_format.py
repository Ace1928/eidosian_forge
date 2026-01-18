import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
def _sympy_format(self, method, variables, backend, default, **kwargs):
    variables = variables or {}
    if backend in (None, math):
        backend = sympy
    variables = defaultkeydict(None if default is None else lambda k: backend.Symbol(default(k)), {k: v if isinstance(v, Expr) else backend.Symbol(v) if isinstance(v, str) else backend.Float(v) for k, v in variables.items()})
    expr = self(variables, backend=backend, **kwargs).simplify()
    if method == 'latex':
        return backend.latex(expr)
    elif method == 'str':
        return str(expr)
    elif method == 'unicode':
        return backend.pretty(expr, use_unicode=True)
    elif method == 'mathml':
        from sympy.printing.mathml import mathml
        return mathml(expr)
    else:
        raise NotImplementedError('Unknown method: %s' % method)