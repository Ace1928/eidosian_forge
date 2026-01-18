from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.polyerrors import (CoercionFailed, NotInvertible,
from sympy.polys.polytools import Poly
from sympy.printing.defaults import DefaultPrinting
class ExtensionElement(DomainElement, DefaultPrinting):
    """
    Element of a finite extension.

    A class of univariate polynomials modulo the ``modulus``
    of the extension ``ext``. It is represented by the
    unique polynomial ``rep`` of lowest degree. Both
    ``rep`` and the representation ``mod`` of ``modulus``
    are of class DMP.

    """
    __slots__ = ('rep', 'ext')

    def __init__(self, rep, ext):
        self.rep = rep
        self.ext = ext

    def parent(f):
        return f.ext

    def __bool__(f):
        return bool(f.rep)

    def __pos__(f):
        return f

    def __neg__(f):
        return ExtElem(-f.rep, f.ext)

    def _get_rep(f, g):
        if isinstance(g, ExtElem):
            if g.ext == f.ext:
                return g.rep
            else:
                return None
        else:
            try:
                g = f.ext.convert(g)
                return g.rep
            except CoercionFailed:
                return None

    def __add__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(f.rep + rep, f.ext)
        else:
            return NotImplemented
    __radd__ = __add__

    def __sub__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(f.rep - rep, f.ext)
        else:
            return NotImplemented

    def __rsub__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(rep - f.rep, f.ext)
        else:
            return NotImplemented

    def __mul__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(f.rep * rep % f.ext.mod, f.ext)
        else:
            return NotImplemented
    __rmul__ = __mul__

    def _divcheck(f):
        """Raise if division is not implemented for this divisor"""
        if not f:
            raise NotInvertible('Zero divisor')
        elif f.ext.is_Field:
            return True
        elif f.rep.is_ground and f.ext.domain.is_unit(f.rep.rep[0]):
            return True
        else:
            msg = f'Can not invert {f} in {f.ext}. Only division by invertible constants is implemented.'
            raise NotImplementedError(msg)

    def inverse(f):
        """Multiplicative inverse.

        Raises
        ======

        NotInvertible
            If the element is a zero divisor.

        """
        f._divcheck()
        if f.ext.is_Field:
            invrep = f.rep.invert(f.ext.mod)
        else:
            R = f.ext.ring
            invrep = R.exquo(R.one, f.rep)
        return ExtElem(invrep, f.ext)

    def __truediv__(f, g):
        rep = f._get_rep(g)
        if rep is None:
            return NotImplemented
        g = ExtElem(rep, f.ext)
        try:
            ginv = g.inverse()
        except NotInvertible:
            raise ZeroDivisionError(f'{f} / {g}')
        return f * ginv
    __floordiv__ = __truediv__

    def __rtruediv__(f, g):
        try:
            g = f.ext.convert(g)
        except CoercionFailed:
            return NotImplemented
        return g / f
    __rfloordiv__ = __rtruediv__

    def __mod__(f, g):
        rep = f._get_rep(g)
        if rep is None:
            return NotImplemented
        g = ExtElem(rep, f.ext)
        try:
            g._divcheck()
        except NotInvertible:
            raise ZeroDivisionError(f'{f} % {g}')
        return f.ext.zero

    def __rmod__(f, g):
        try:
            g = f.ext.convert(g)
        except CoercionFailed:
            return NotImplemented
        return g % f

    def __pow__(f, n):
        if not isinstance(n, int):
            raise TypeError("exponent of type 'int' expected")
        if n < 0:
            try:
                f, n = (f.inverse(), -n)
            except NotImplementedError:
                raise ValueError('negative powers are not defined')
        b = f.rep
        m = f.ext.mod
        r = f.ext.one.rep
        while n > 0:
            if n % 2:
                r = r * b % m
            b = b * b % m
            n //= 2
        return ExtElem(r, f.ext)

    def __eq__(f, g):
        if isinstance(g, ExtElem):
            return f.rep == g.rep and f.ext == g.ext
        else:
            return NotImplemented

    def __ne__(f, g):
        return not f == g

    def __hash__(f):
        return hash((f.rep, f.ext))

    def __str__(f):
        from sympy.printing.str import sstr
        return sstr(f.rep)
    __repr__ = __str__

    @property
    def is_ground(f):
        return f.rep.is_ground

    def to_ground(f):
        [c] = f.rep.to_list()
        return c