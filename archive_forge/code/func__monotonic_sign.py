from .add import Add
from .mul import Mul, _keep_coeff
from .power import Pow
from .basic import Basic
from .expr import Expr
from .function import expand_power_exp
from .sympify import sympify
from .numbers import Rational, Integer, Number, I, equal_valued
from .singleton import S
from .sorting import default_sort_key, ordered
from .symbol import Dummy
from .traversal import preorder_traversal
from .coreerrors import NonCommutativeExpression
from .containers import Tuple, Dict
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import (common_prefix, common_suffix,
from collections import defaultdict
from typing import Tuple as tTuple
def _monotonic_sign(self):
    """Return the value closest to 0 that ``self`` may have if all symbols
    are signed and the result is uniformly the same sign for all values of symbols.
    If a symbol is only signed but not known to be an
    integer or the result is 0 then a symbol representative of the sign of self
    will be returned. Otherwise, None is returned if a) the sign could be positive
    or negative or b) self is not in one of the following forms:

    - L(x, y, ...) + A: a function linear in all symbols x, y, ... with an
      additive constant; if A is zero then the function can be a monomial whose
      sign is monotonic over the range of the variables, e.g. (x + 1)**3 if x is
      nonnegative.
    - A/L(x, y, ...) + B: the inverse of a function linear in all symbols x, y, ...
      that does not have a sign change from positive to negative for any set
      of values for the variables.
    - M(x, y, ...) + A: a monomial M whose factors are all signed and a constant, A.
    - A/M(x, y, ...) + B: the inverse of a monomial and constants A and B.
    - P(x): a univariate polynomial

    Examples
    ========

    >>> from sympy.core.exprtools import _monotonic_sign as F
    >>> from sympy import Dummy
    >>> nn = Dummy(integer=True, nonnegative=True)
    >>> p = Dummy(integer=True, positive=True)
    >>> p2 = Dummy(integer=True, positive=True)
    >>> F(nn + 1)
    1
    >>> F(p - 1)
    _nneg
    >>> F(nn*p + 1)
    1
    >>> F(p2*p + 1)
    2
    >>> F(nn - 1)  # could be negative, zero or positive
    """
    if not self.is_extended_real:
        return
    if (-self).is_Symbol:
        rv = _monotonic_sign(-self)
        return rv if rv is None else -rv
    if not self.is_Add and self.as_numer_denom()[1].is_number:
        s = self
        if s.is_prime:
            if s.is_odd:
                return Integer(3)
            else:
                return Integer(2)
        elif s.is_composite:
            if s.is_odd:
                return Integer(9)
            else:
                return Integer(4)
        elif s.is_positive:
            if s.is_even:
                if s.is_prime is False:
                    return Integer(4)
                else:
                    return Integer(2)
            elif s.is_integer:
                return S.One
            else:
                return _eps
        elif s.is_extended_negative:
            if s.is_even:
                return Integer(-2)
            elif s.is_integer:
                return S.NegativeOne
            else:
                return -_eps
        if s.is_zero or s.is_extended_nonpositive or s.is_extended_nonnegative:
            return S.Zero
        return None
    free = self.free_symbols
    if len(free) == 1:
        if self.is_polynomial():
            from sympy.polys.polytools import real_roots
            from sympy.polys.polyroots import roots
            from sympy.polys.polyerrors import PolynomialError
            x = free.pop()
            x0 = _monotonic_sign(x)
            if x0 in (_eps, -_eps):
                x0 = S.Zero
            if x0 is not None:
                d = self.diff(x)
                if d.is_number:
                    currentroots = []
                else:
                    try:
                        currentroots = real_roots(d)
                    except (PolynomialError, NotImplementedError):
                        currentroots = [r for r in roots(d, x) if r.is_extended_real]
                y = self.subs(x, x0)
                if x.is_nonnegative and all(((r - x0).is_nonpositive for r in currentroots)):
                    if y.is_nonnegative and d.is_positive:
                        if y:
                            return y if y.is_positive else Dummy('pos', positive=True)
                        else:
                            return Dummy('nneg', nonnegative=True)
                    if y.is_nonpositive and d.is_negative:
                        if y:
                            return y if y.is_negative else Dummy('neg', negative=True)
                        else:
                            return Dummy('npos', nonpositive=True)
                elif x.is_nonpositive and all(((r - x0).is_nonnegative for r in currentroots)):
                    if y.is_nonnegative and d.is_negative:
                        if y:
                            return Dummy('pos', positive=True)
                        else:
                            return Dummy('nneg', nonnegative=True)
                    if y.is_nonpositive and d.is_positive:
                        if y:
                            return Dummy('neg', negative=True)
                        else:
                            return Dummy('npos', nonpositive=True)
        else:
            n, d = self.as_numer_denom()
            den = None
            if n.is_number:
                den = _monotonic_sign(d)
            elif not d.is_number:
                if _monotonic_sign(n) is not None:
                    den = _monotonic_sign(d)
            if den is not None and (den.is_positive or den.is_negative):
                v = n * den
                if v.is_positive:
                    return Dummy('pos', positive=True)
                elif v.is_nonnegative:
                    return Dummy('nneg', nonnegative=True)
                elif v.is_negative:
                    return Dummy('neg', negative=True)
                elif v.is_nonpositive:
                    return Dummy('npos', nonpositive=True)
        return None
    c, a = self.as_coeff_Add()
    v = None
    if not a.is_polynomial():
        n, d = a.as_numer_denom()
        if not (n.is_number or d.is_number):
            return
        if (a.is_Mul or a.is_Pow) and a.is_rational and all((p.exp.is_Integer for p in a.atoms(Pow) if p.is_Pow)) and (a.is_positive or a.is_negative):
            v = S.One
            for ai in Mul.make_args(a):
                if ai.is_number:
                    v *= ai
                    continue
                reps = {}
                for x in ai.free_symbols:
                    reps[x] = _monotonic_sign(x)
                    if reps[x] is None:
                        return
                v *= ai.subs(reps)
    elif c:
        if not any((p for p in a.atoms(Pow) if not p.is_number)) and (a.is_nonpositive or a.is_nonnegative):
            free = list(a.free_symbols)
            p = {}
            for i in free:
                v = _monotonic_sign(i)
                if v is None:
                    return
                p[i] = v or (_eps if i.is_nonnegative else -_eps)
            v = a.xreplace(p)
    if v is not None:
        rv = v + c
        if v.is_nonnegative and rv.is_positive:
            return rv.subs(_eps, 0)
        if v.is_nonpositive and rv.is_negative:
            return rv.subs(_eps, 0)