from __future__ import annotations
from typing import Callable
from math import log as _log, sqrt as _sqrt
from itertools import product
from .sympify import _sympify
from .cache import cacheit
from .singleton import S
from .expr import Expr
from .evalf import PrecisionExhausted
from .function import (expand_complex, expand_multinomial,
from .logic import fuzzy_bool, fuzzy_not, fuzzy_and, fuzzy_or
from .parameters import global_parameters
from .relational import is_gt, is_lt
from .kind import NumberKind, UndefinedKind
from sympy.external.gmpy import HAS_GMPY, gmpy
from sympy.utilities.iterables import sift
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int
from sympy.multipledispatch import Dispatcher
from mpmath.libmp import sqrtrem as mpmath_sqrtrem
from .add import Add
from .numbers import Integer
from .mul import Mul, _keep_coeff
from .symbol import Symbol, Dummy, symbols
def _eval_expand_multinomial(self, **hints):
    """(a + b + ..)**n -> a**n + n*a**(n-1)*b + .., n is nonzero integer"""
    base, exp = self.args
    result = self
    if exp.is_Rational and exp.p > 0 and base.is_Add:
        if not exp.is_Integer:
            n = Integer(exp.p // exp.q)
            if not n:
                return result
            else:
                radical, result = (self.func(base, exp - n), [])
                expanded_base_n = self.func(base, n)
                if expanded_base_n.is_Pow:
                    expanded_base_n = expanded_base_n._eval_expand_multinomial()
                for term in Add.make_args(expanded_base_n):
                    result.append(term * radical)
                return Add(*result)
        n = int(exp)
        if base.is_commutative:
            order_terms, other_terms = ([], [])
            for b in base.args:
                if b.is_Order:
                    order_terms.append(b)
                else:
                    other_terms.append(b)
            if order_terms:
                f = Add(*other_terms)
                o = Add(*order_terms)
                if n == 2:
                    return expand_multinomial(f ** n, deep=False) + n * f * o
                else:
                    g = expand_multinomial(f ** (n - 1), deep=False)
                    return expand_mul(f * g, deep=False) + n * g * o
            if base.is_number:
                a, b = base.as_real_imag()
                if a.is_Rational and b.is_Rational:
                    if not a.is_Integer:
                        if not b.is_Integer:
                            k = self.func(a.q * b.q, n)
                            a, b = (a.p * b.q, a.q * b.p)
                        else:
                            k = self.func(a.q, n)
                            a, b = (a.p, a.q * b)
                    elif not b.is_Integer:
                        k = self.func(b.q, n)
                        a, b = (a * b.q, b.p)
                    else:
                        k = 1
                    a, b, c, d = (int(a), int(b), 1, 0)
                    while n:
                        if n & 1:
                            c, d = (a * c - b * d, b * c + a * d)
                            n -= 1
                        a, b = (a * a - b * b, 2 * a * b)
                        n //= 2
                    I = S.ImaginaryUnit
                    if k == 1:
                        return c + I * d
                    else:
                        return Integer(c) / k + I * d / k
            p = other_terms
            from sympy.ntheory.multinomial import multinomial_coefficients
            from sympy.polys.polyutils import basic_from_dict
            expansion_dict = multinomial_coefficients(len(p), n)
            return basic_from_dict(expansion_dict, *p)
        elif n == 2:
            return Add(*[f * g for f in base.args for g in base.args])
        else:
            multi = (base ** (n - 1))._eval_expand_multinomial()
            if multi.is_Add:
                return Add(*[f * g for f in base.args for g in multi.args])
            else:
                return Add(*[f * multi for f in base.args])
    elif exp.is_Rational and exp.p < 0 and base.is_Add and (abs(exp.p) > exp.q):
        return 1 / self.func(base, -exp)._eval_expand_multinomial()
    elif exp.is_Add and base.is_Number and (hints.get('force', False) or base.is_zero is False or exp._all_nonneg_or_nonppos()):
        coeff, tail = ([], [])
        for term in exp.args:
            if term.is_Number:
                coeff.append(self.func(base, term))
            else:
                tail.append(term)
        return Mul(*coeff + [self.func(base, Add._from_args(tail))])
    else:
        return result