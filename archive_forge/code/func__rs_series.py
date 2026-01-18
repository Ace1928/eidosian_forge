from sympy.polys.domains import QQ, EX
from sympy.polys.rings import PolyElement, ring, sring
from sympy.polys.polyerrors import DomainError
from sympy.polys.monomials import (monomial_min, monomial_mul, monomial_div,
from mpmath.libmp.libintmath import ifac
from sympy.core import PoleError, Function, Expr
from sympy.core.numbers import Rational, igcd
from sympy.functions import sin, cos, tan, atan, exp, atanh, tanh, log, ceiling
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import giant_steps
import math
def _rs_series(expr, series_rs, a, prec):
    args = expr.args
    R = series_rs.ring
    if not any((arg.has(Function) for arg in args)) and (not expr.is_Function):
        return series_rs
    if not expr.has(a):
        return series_rs
    elif expr.is_Function:
        arg = args[0]
        if len(args) > 1:
            raise NotImplementedError
        R1, series = sring(arg, domain=QQ, expand=False, series=True)
        series_inner = _rs_series(arg, series, a, prec)
        R = R.compose(R1).compose(series_inner.ring)
        series_inner = series_inner.set_ring(R)
        series = eval(_convert_func[str(expr.func)])(series_inner, R(a), prec)
        return series
    elif expr.is_Mul:
        n = len(args)
        for arg in args:
            if not arg.is_Number:
                R1, _ = sring(arg, expand=False, series=True)
                R = R.compose(R1)
        min_pows = list(map(rs_min_pow, args, [R(arg) for arg in args], [a] * len(args)))
        sum_pows = sum(min_pows)
        series = R(1)
        for i in range(n):
            _series = _rs_series(args[i], R(args[i]), a, prec - sum_pows + min_pows[i])
            R = R.compose(_series.ring)
            _series = _series.set_ring(R)
            series = series.set_ring(R)
            series *= _series
        series = rs_trunc(series, R(a), prec)
        return series
    elif expr.is_Add:
        n = len(args)
        series = R(0)
        for i in range(n):
            _series = _rs_series(args[i], R(args[i]), a, prec)
            R = R.compose(_series.ring)
            _series = _series.set_ring(R)
            series = series.set_ring(R)
            series += _series
        return series
    elif expr.is_Pow:
        R1, _ = sring(expr.base, domain=QQ, expand=False, series=True)
        R = R.compose(R1)
        series_inner = _rs_series(expr.base, R(expr.base), a, prec)
        return rs_pow(series_inner, expr.exp, series_inner.ring(a), prec)
    elif isinstance(expr, Expr) and expr.is_constant():
        return sring(expr, domain=QQ, expand=False, series=True)[1]
    else:
        raise NotImplementedError