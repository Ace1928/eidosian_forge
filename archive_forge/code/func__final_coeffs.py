from sympy.core import S, sympify
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int
def _final_coeffs(n):
    if n < k:
        return [S.Zero] * n + [S.One] + [S.Zero] * (k - n - 1)
    else:
        return _square_and_reduce(_final_coeffs(n // 2), n % 2)