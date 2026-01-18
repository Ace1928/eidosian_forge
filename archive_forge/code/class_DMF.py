from sympy.core.numbers import oo
from sympy.core.sympify import CantSympify
from sympy.polys.polyerrors import CoercionFailed, NotReversible, NotInvertible
from sympy.polys.polyutils import PicklableWithSlots
from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.sqfreetools import (
from sympy.polys.factortools import (
from sympy.polys.rootisolation import (
from sympy.polys.polyerrors import (
class DMF(PicklableWithSlots, CantSympify):
    """Dense Multivariate Fractions over `K`. """
    __slots__ = ('num', 'den', 'lev', 'dom', 'ring')

    def __init__(self, rep, dom, lev=None, ring=None):
        num, den, lev = self._parse(rep, dom, lev)
        num, den = dmp_cancel(num, den, lev, dom)
        self.num = num
        self.den = den
        self.lev = lev
        self.dom = dom
        self.ring = ring

    @classmethod
    def new(cls, rep, dom, lev=None, ring=None):
        num, den, lev = cls._parse(rep, dom, lev)
        obj = object.__new__(cls)
        obj.num = num
        obj.den = den
        obj.lev = lev
        obj.dom = dom
        obj.ring = ring
        return obj

    @classmethod
    def _parse(cls, rep, dom, lev=None):
        if isinstance(rep, tuple):
            num, den = rep
            if lev is not None:
                if isinstance(num, dict):
                    num = dmp_from_dict(num, lev, dom)
                if isinstance(den, dict):
                    den = dmp_from_dict(den, lev, dom)
            else:
                num, num_lev = dmp_validate(num)
                den, den_lev = dmp_validate(den)
                if num_lev == den_lev:
                    lev = num_lev
                else:
                    raise ValueError('inconsistent number of levels')
            if dmp_zero_p(den, lev):
                raise ZeroDivisionError('fraction denominator')
            if dmp_zero_p(num, lev):
                den = dmp_one(lev, dom)
            elif dmp_negative_p(den, lev, dom):
                num = dmp_neg(num, lev, dom)
                den = dmp_neg(den, lev, dom)
        else:
            num = rep
            if lev is not None:
                if isinstance(num, dict):
                    num = dmp_from_dict(num, lev, dom)
                elif not isinstance(num, list):
                    num = dmp_ground(dom.convert(num), lev)
            else:
                num, lev = dmp_validate(num)
            den = dmp_one(lev, dom)
        return (num, den, lev)

    def __repr__(f):
        return '%s((%s, %s), %s, %s)' % (f.__class__.__name__, f.num, f.den, f.dom, f.ring)

    def __hash__(f):
        return hash((f.__class__.__name__, dmp_to_tuple(f.num, f.lev), dmp_to_tuple(f.den, f.lev), f.lev, f.dom, f.ring))

    def poly_unify(f, g):
        """Unify a multivariate fraction and a polynomial. """
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed('Cannot unify %s with %s' % (f, g))
        if f.dom == g.dom and f.ring == g.ring:
            return (f.lev, f.dom, f.per, (f.num, f.den), g.rep)
        else:
            lev, dom = (f.lev, f.dom.unify(g.dom))
            ring = f.ring
            if g.ring is not None:
                if ring is not None:
                    ring = ring.unify(g.ring)
                else:
                    ring = g.ring
            F = (dmp_convert(f.num, lev, f.dom, dom), dmp_convert(f.den, lev, f.dom, dom))
            G = dmp_convert(g.rep, lev, g.dom, dom)

            def per(num, den, cancel=True, kill=False, lev=lev):
                if kill:
                    if not lev:
                        return num / den
                    else:
                        lev = lev - 1
                if cancel:
                    num, den = dmp_cancel(num, den, lev, dom)
                return f.__class__.new((num, den), dom, lev, ring=ring)
            return (lev, dom, per, F, G)

    def frac_unify(f, g):
        """Unify representations of two multivariate fractions. """
        if not isinstance(g, DMF) or f.lev != g.lev:
            raise UnificationFailed('Cannot unify %s with %s' % (f, g))
        if f.dom == g.dom and f.ring == g.ring:
            return (f.lev, f.dom, f.per, (f.num, f.den), (g.num, g.den))
        else:
            lev, dom = (f.lev, f.dom.unify(g.dom))
            ring = f.ring
            if g.ring is not None:
                if ring is not None:
                    ring = ring.unify(g.ring)
                else:
                    ring = g.ring
            F = (dmp_convert(f.num, lev, f.dom, dom), dmp_convert(f.den, lev, f.dom, dom))
            G = (dmp_convert(g.num, lev, g.dom, dom), dmp_convert(g.den, lev, g.dom, dom))

            def per(num, den, cancel=True, kill=False, lev=lev):
                if kill:
                    if not lev:
                        return num / den
                    else:
                        lev = lev - 1
                if cancel:
                    num, den = dmp_cancel(num, den, lev, dom)
                return f.__class__.new((num, den), dom, lev, ring=ring)
            return (lev, dom, per, F, G)

    def per(f, num, den, cancel=True, kill=False, ring=None):
        """Create a DMF out of the given representation. """
        lev, dom = (f.lev, f.dom)
        if kill:
            if not lev:
                return num / den
            else:
                lev -= 1
        if cancel:
            num, den = dmp_cancel(num, den, lev, dom)
        if ring is None:
            ring = f.ring
        return f.__class__.new((num, den), dom, lev, ring=ring)

    def half_per(f, rep, kill=False):
        """Create a DMP out of the given representation. """
        lev = f.lev
        if kill:
            if not lev:
                return rep
            else:
                lev -= 1
        return DMP(rep, f.dom, lev)

    @classmethod
    def zero(cls, lev, dom, ring=None):
        return cls.new(0, dom, lev, ring=ring)

    @classmethod
    def one(cls, lev, dom, ring=None):
        return cls.new(1, dom, lev, ring=ring)

    def numer(f):
        """Returns the numerator of ``f``. """
        return f.half_per(f.num)

    def denom(f):
        """Returns the denominator of ``f``. """
        return f.half_per(f.den)

    def cancel(f):
        """Remove common factors from ``f.num`` and ``f.den``. """
        return f.per(f.num, f.den)

    def neg(f):
        """Negate all coefficients in ``f``. """
        return f.per(dmp_neg(f.num, f.lev, f.dom), f.den, cancel=False)

    def add(f, g):
        """Add two multivariate fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = (dmp_add_mul(F_num, F_den, G, lev, dom), F_den)
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = (F, G)
            num = dmp_add(dmp_mul(F_num, G_den, lev, dom), dmp_mul(F_den, G_num, lev, dom), lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)
        return per(num, den)

    def sub(f, g):
        """Subtract two multivariate fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = (dmp_sub_mul(F_num, F_den, G, lev, dom), F_den)
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = (F, G)
            num = dmp_sub(dmp_mul(F_num, G_den, lev, dom), dmp_mul(F_den, G_num, lev, dom), lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)
        return per(num, den)

    def mul(f, g):
        """Multiply two multivariate fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = (dmp_mul(F_num, G, lev, dom), F_den)
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = (F, G)
            num = dmp_mul(F_num, G_num, lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)
        return per(num, den)

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if isinstance(n, int):
            num, den = (f.num, f.den)
            if n < 0:
                num, den, n = (den, num, -n)
            return f.per(dmp_pow(num, n, f.lev, f.dom), dmp_pow(den, n, f.lev, f.dom), cancel=False)
        else:
            raise TypeError('``int`` expected, got %s' % type(n))

    def quo(f, g):
        """Computes quotient of fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = (F_num, dmp_mul(F_den, G, lev, dom))
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = (F, G)
            num = dmp_mul(F_num, G_den, lev, dom)
            den = dmp_mul(F_den, G_num, lev, dom)
        res = per(num, den)
        if f.ring is not None and res not in f.ring:
            from sympy.polys.polyerrors import ExactQuotientFailed
            raise ExactQuotientFailed(f, g, f.ring)
        return res
    exquo = quo

    def invert(f, check=True):
        """Computes inverse of a fraction ``f``. """
        if check and f.ring is not None and (not f.ring.is_unit(f)):
            raise NotReversible(f, f.ring)
        res = f.per(f.den, f.num, cancel=False)
        return res

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero fraction. """
        return dmp_zero_p(f.num, f.lev)

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit fraction. """
        return dmp_one_p(f.num, f.lev, f.dom) and dmp_one_p(f.den, f.lev, f.dom)

    def __neg__(f):
        return f.neg()

    def __add__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.add(g)
        try:
            return f.add(f.half_per(g))
        except TypeError:
            return NotImplemented
        except (CoercionFailed, NotImplementedError):
            if f.ring is not None:
                try:
                    return f.add(f.ring.convert(g))
                except (CoercionFailed, NotImplementedError):
                    pass
            return NotImplemented

    def __radd__(f, g):
        return f.__add__(g)

    def __sub__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.sub(g)
        try:
            return f.sub(f.half_per(g))
        except TypeError:
            return NotImplemented
        except (CoercionFailed, NotImplementedError):
            if f.ring is not None:
                try:
                    return f.sub(f.ring.convert(g))
                except (CoercionFailed, NotImplementedError):
                    pass
            return NotImplemented

    def __rsub__(f, g):
        return (-f).__add__(g)

    def __mul__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.mul(g)
        try:
            return f.mul(f.half_per(g))
        except TypeError:
            return NotImplemented
        except (CoercionFailed, NotImplementedError):
            if f.ring is not None:
                try:
                    return f.mul(f.ring.convert(g))
                except (CoercionFailed, NotImplementedError):
                    pass
            return NotImplemented

    def __rmul__(f, g):
        return f.__mul__(g)

    def __pow__(f, n):
        return f.pow(n)

    def __truediv__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.quo(g)
        try:
            return f.quo(f.half_per(g))
        except TypeError:
            return NotImplemented
        except (CoercionFailed, NotImplementedError):
            if f.ring is not None:
                try:
                    return f.quo(f.ring.convert(g))
                except (CoercionFailed, NotImplementedError):
                    pass
            return NotImplemented

    def __rtruediv__(self, g):
        r = self.invert(check=False) * g
        if self.ring and r not in self.ring:
            from sympy.polys.polyerrors import ExactQuotientFailed
            raise ExactQuotientFailed(g, self, self.ring)
        return r

    def __eq__(f, g):
        try:
            if isinstance(g, DMP):
                _, _, _, (F_num, F_den), G = f.poly_unify(g)
                if f.lev == g.lev:
                    return dmp_one_p(F_den, f.lev, f.dom) and F_num == G
            else:
                _, _, _, F, G = f.frac_unify(g)
                if f.lev == g.lev:
                    return F == G
        except UnificationFailed:
            pass
        return False

    def __ne__(f, g):
        try:
            if isinstance(g, DMP):
                _, _, _, (F_num, F_den), G = f.poly_unify(g)
                if f.lev == g.lev:
                    return not (dmp_one_p(F_den, f.lev, f.dom) and F_num == G)
            else:
                _, _, _, F, G = f.frac_unify(g)
                if f.lev == g.lev:
                    return F != G
        except UnificationFailed:
            pass
        return True

    def __lt__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F < G

    def __le__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F <= G

    def __gt__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F > G

    def __ge__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F >= G

    def __bool__(f):
        return not dmp_zero_p(f.num, f.lev)