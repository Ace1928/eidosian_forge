from sympy.core import Add, Expr, Mul, S, sympify
from sympy.core.function import _mexpand, count_ops, expand_mul
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import root, sign, sqrt
from sympy.polys import Poly, PolynomialError
def _sqrt_numeric_denest(a, b, r, d2):
    """Helper that denest
    $\\sqrt{a + b \\sqrt{r}}, d^2 = a^2 - b^2 r > 0$

    If it cannot be denested, it returns ``None``.
    """
    d = sqrt(d2)
    s = a + d
    if sqrt_depth(s) < sqrt_depth(r) + 1 or (s ** 2).is_Rational:
        s1, s2 = (sign(s), sign(b))
        if s1 == s2 == -1:
            s1 = s2 = 1
        res = (s1 * sqrt(a + d) + s2 * sqrt(a - d)) * sqrt(2) / 2
        return res.expand()