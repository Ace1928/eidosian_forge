from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options
import re
def _nsort(roots, separated=False):
    """Sort the numerical roots putting the real roots first, then sorting
    according to real and imaginary parts. If ``separated`` is True, then
    the real and imaginary roots will be returned in two lists, respectively.

    This routine tries to avoid issue 6137 by separating the roots into real
    and imaginary parts before evaluation. In addition, the sorting will raise
    an error if any computation cannot be done with precision.
    """
    if not all((r.is_number for r in roots)):
        raise NotImplementedError
    key = [[i.n(2).as_real_imag()[0] for i in r.as_real_imag()] for r in roots]
    if len(roots) > 1 and any((i._prec == 1 for k in key for i in k)):
        raise NotImplementedError('could not compute root with precision')
    key = [(1 if i else 0, r, i) for r, i in key]
    key = sorted(zip(key, roots))
    if separated:
        r = []
        i = []
        for (im, _, _), v in key:
            if im:
                i.append(v)
            else:
                r.append(v)
        return (r, i)
    _, roots = zip(*key)
    return list(roots)