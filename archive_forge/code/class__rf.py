from sympy.core import Function, S, Mul, Pow, Add
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.function import expand_func
from sympy.core.symbol import Dummy
from sympy.functions import gamma, sqrt, sin
from sympy.polys import factor, cancel
from sympy.utilities.iterables import sift, uniq
class _rf(Function):

    @classmethod
    def eval(cls, a, b):
        if b.is_Integer:
            if not b:
                return S.One
            n = int(b)
            if n > 0:
                return Mul(*[a + i for i in range(n)])
            elif n < 0:
                return 1 / Mul(*[a - i for i in range(1, -n + 1)])
        else:
            if b.is_Add:
                c, _b = b.as_coeff_Add()
                if c.is_Integer:
                    if c > 0:
                        return _rf(a, _b) * _rf(a + _b, c)
                    elif c < 0:
                        return _rf(a, _b) / _rf(a + _b + c, -c)
            if a.is_Add:
                c, _a = a.as_coeff_Add()
                if c.is_Integer:
                    if c > 0:
                        return _rf(_a, b) * _rf(_a + b, c) / _rf(_a, c)
                    elif c < 0:
                        return _rf(_a, b) * _rf(_a + c, -c) / _rf(_a + b + c, -c)