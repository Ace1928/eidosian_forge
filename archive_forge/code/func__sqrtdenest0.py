from sympy.core import Add, Expr, Mul, S, sympify
from sympy.core.function import _mexpand, count_ops, expand_mul
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import root, sign, sqrt
from sympy.polys import Poly, PolynomialError
def _sqrtdenest0(expr):
    """Returns expr after denesting its arguments."""
    if is_sqrt(expr):
        n, d = expr.as_numer_denom()
        if d is S.One:
            if n.base.is_Add:
                args = sorted(n.base.args, key=default_sort_key)
                if len(args) > 2 and all(((x ** 2).is_Integer for x in args)):
                    try:
                        return _sqrtdenest_rec(n)
                    except SqrtdenestStopIteration:
                        pass
                expr = sqrt(_mexpand(Add(*[_sqrtdenest0(x) for x in args])))
            return _sqrtdenest1(expr)
        else:
            n, d = [_sqrtdenest0(i) for i in (n, d)]
            return n / d
    if isinstance(expr, Add):
        cs = []
        args = []
        for arg in expr.args:
            c, a = arg.as_coeff_Mul()
            cs.append(c)
            args.append(a)
        if all((c.is_Rational for c in cs)) and all((is_sqrt(arg) for arg in args)):
            return _sqrt_ratcomb(cs, args)
    if isinstance(expr, Expr):
        args = expr.args
        if args:
            return expr.func(*[_sqrtdenest0(a) for a in args])
    return expr