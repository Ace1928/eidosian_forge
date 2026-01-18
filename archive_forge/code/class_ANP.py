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
class ANP(PicklableWithSlots, CantSympify):
    """Dense Algebraic Number Polynomials over a field. """
    __slots__ = ('rep', 'mod', 'dom')

    def __init__(self, rep, mod, dom):
        if type(rep) is dict:
            self.rep = dup_from_dict(rep, dom)
        else:
            if isinstance(rep, list):
                rep = [dom.convert(a) for a in rep]
            else:
                rep = [dom.convert(rep)]
            self.rep = dup_strip(rep)
        if isinstance(mod, DMP):
            self.mod = mod.rep
        elif isinstance(mod, dict):
            self.mod = dup_from_dict(mod, dom)
        else:
            self.mod = dup_strip(mod)
        self.dom = dom

    def __repr__(f):
        return '%s(%s, %s, %s)' % (f.__class__.__name__, f.rep, f.mod, f.dom)

    def __hash__(f):
        return hash((f.__class__.__name__, f.to_tuple(), dmp_to_tuple(f.mod, 0), f.dom))

    def unify(f, g):
        """Unify representations of two algebraic numbers. """
        if not isinstance(g, ANP) or f.mod != g.mod:
            raise UnificationFailed('Cannot unify %s with %s' % (f, g))
        if f.dom == g.dom:
            return (f.dom, f.per, f.rep, g.rep, f.mod)
        else:
            dom = f.dom.unify(g.dom)
            F = dup_convert(f.rep, f.dom, dom)
            G = dup_convert(g.rep, g.dom, dom)
            if dom != f.dom and dom != g.dom:
                mod = dup_convert(f.mod, f.dom, dom)
            elif dom == f.dom:
                mod = f.mod
            else:
                mod = g.mod
            per = lambda rep: ANP(rep, mod, dom)
        return (dom, per, F, G, mod)

    def per(f, rep, mod=None, dom=None):
        return ANP(rep, mod or f.mod, dom or f.dom)

    @classmethod
    def zero(cls, mod, dom):
        return ANP(0, mod, dom)

    @classmethod
    def one(cls, mod, dom):
        return ANP(1, mod, dom)

    def to_dict(f):
        """Convert ``f`` to a dict representation with native coefficients. """
        return dmp_to_dict(f.rep, 0, f.dom)

    def to_sympy_dict(f):
        """Convert ``f`` to a dict representation with SymPy coefficients. """
        rep = dmp_to_dict(f.rep, 0, f.dom)
        for k, v in rep.items():
            rep[k] = f.dom.to_sympy(v)
        return rep

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        return f.rep

    def to_sympy_list(f):
        """Convert ``f`` to a list representation with SymPy coefficients. """
        return [f.dom.to_sympy(c) for c in f.rep]

    def to_tuple(f):
        """
        Convert ``f`` to a tuple representation with native coefficients.

        This is needed for hashing.
        """
        return dmp_to_tuple(f.rep, 0)

    @classmethod
    def from_list(cls, rep, mod, dom):
        return ANP(dup_strip(list(map(dom.convert, rep))), mod, dom)

    def neg(f):
        return f.per(dup_neg(f.rep, f.dom))

    def add(f, g):
        dom, per, F, G, mod = f.unify(g)
        return per(dup_add(F, G, dom))

    def sub(f, g):
        dom, per, F, G, mod = f.unify(g)
        return per(dup_sub(F, G, dom))

    def mul(f, g):
        dom, per, F, G, mod = f.unify(g)
        return per(dup_rem(dup_mul(F, G, dom), mod, dom))

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if isinstance(n, int):
            if n < 0:
                F, n = (dup_invert(f.rep, f.mod, f.dom), -n)
            else:
                F = f.rep
            return f.per(dup_rem(dup_pow(F, n, f.dom), f.mod, f.dom))
        else:
            raise TypeError('``int`` expected, got %s' % type(n))

    def div(f, g):
        dom, per, F, G, mod = f.unify(g)
        return (per(dup_rem(dup_mul(F, dup_invert(G, mod, dom), dom), mod, dom)), f.zero(mod, dom))

    def rem(f, g):
        dom, _, _, G, mod = f.unify(g)
        s, h = dup_half_gcdex(G, mod, dom)
        if h == [dom.one]:
            return f.zero(mod, dom)
        else:
            raise NotInvertible('zero divisor')

    def quo(f, g):
        dom, per, F, G, mod = f.unify(g)
        return per(dup_rem(dup_mul(F, dup_invert(G, mod, dom), dom), mod, dom))
    exquo = quo

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        return dup_LC(f.rep, f.dom)

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        return dup_TC(f.rep, f.dom)

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero algebraic number. """
        return not f

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit algebraic number. """
        return f.rep == [f.dom.one]

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        return not f.rep or len(f.rep) == 1

    def __pos__(f):
        return f

    def __neg__(f):
        return f.neg()

    def __add__(f, g):
        if isinstance(g, ANP):
            return f.add(g)
        else:
            try:
                return f.add(f.per(g))
            except (CoercionFailed, TypeError):
                return NotImplemented

    def __radd__(f, g):
        return f.__add__(g)

    def __sub__(f, g):
        if isinstance(g, ANP):
            return f.sub(g)
        else:
            try:
                return f.sub(f.per(g))
            except (CoercionFailed, TypeError):
                return NotImplemented

    def __rsub__(f, g):
        return (-f).__add__(g)

    def __mul__(f, g):
        if isinstance(g, ANP):
            return f.mul(g)
        else:
            try:
                return f.mul(f.per(g))
            except (CoercionFailed, TypeError):
                return NotImplemented

    def __rmul__(f, g):
        return f.__mul__(g)

    def __pow__(f, n):
        return f.pow(n)

    def __divmod__(f, g):
        return f.div(g)

    def __mod__(f, g):
        return f.rem(g)

    def __truediv__(f, g):
        if isinstance(g, ANP):
            return f.quo(g)
        else:
            try:
                return f.quo(f.per(g))
            except (CoercionFailed, TypeError):
                return NotImplemented

    def __eq__(f, g):
        try:
            _, _, F, G, _ = f.unify(g)
            return F == G
        except UnificationFailed:
            return False

    def __ne__(f, g):
        try:
            _, _, F, G, _ = f.unify(g)
            return F != G
        except UnificationFailed:
            return True

    def __lt__(f, g):
        _, _, F, G, _ = f.unify(g)
        return F < G

    def __le__(f, g):
        _, _, F, G, _ = f.unify(g)
        return F <= G

    def __gt__(f, g):
        _, _, F, G, _ = f.unify(g)
        return F > G

    def __ge__(f, g):
        _, _, F, G, _ = f.unify(g)
        return F >= G

    def __bool__(f):
        return bool(f.rep)