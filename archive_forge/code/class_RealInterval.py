from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
class RealInterval:
    """A fully qualified representation of a real isolation interval. """

    def __init__(self, data, f, dom):
        """Initialize new real interval with complete information. """
        if len(data) == 2:
            s, t = data
            self.neg = False
            if s < 0:
                if t <= 0:
                    f, s, t, self.neg = (dup_mirror(f, dom), -t, -s, True)
                else:
                    raise ValueError('Cannot refine a real root in (%s, %s)' % (s, t))
            a, b, c, d = _mobius_from_interval((s, t), dom.get_field())
            f = dup_transform(f, dup_strip([a, b]), dup_strip([c, d]), dom)
            self.mobius = (a, b, c, d)
        else:
            self.mobius = data[:-1]
            self.neg = data[-1]
        self.f, self.dom = (f, dom)

    @property
    def func(self):
        return RealInterval

    @property
    def args(self):
        i = self
        return (i.mobius + (i.neg,), i.f, i.dom)

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        return self.args == other.args

    @property
    def a(self):
        """Return the position of the left end. """
        field = self.dom.get_field()
        a, b, c, d = self.mobius
        if not self.neg:
            if a * d < b * c:
                return field(a, c)
            return field(b, d)
        else:
            if a * d > b * c:
                return -field(a, c)
            return -field(b, d)

    @property
    def b(self):
        """Return the position of the right end. """
        was = self.neg
        self.neg = not was
        rv = -self.a
        self.neg = was
        return rv

    @property
    def dx(self):
        """Return width of the real isolating interval. """
        return self.b - self.a

    @property
    def center(self):
        """Return the center of the real isolating interval. """
        return (self.a + self.b) / 2

    @property
    def max_denom(self):
        """Return the largest denominator occurring in either endpoint. """
        return max(self.a.denominator, self.b.denominator)

    def as_tuple(self):
        """Return tuple representation of real isolating interval. """
        return (self.a, self.b)

    def __repr__(self):
        return '(%s, %s)' % (self.a, self.b)

    def __contains__(self, item):
        """
        Say whether a complex number belongs to this real interval.

        Parameters
        ==========

        item : pair (re, im) or number re
            Either a pair giving the real and imaginary parts of the number,
            or else a real number.

        """
        if isinstance(item, tuple):
            re, im = item
        else:
            re, im = (item, 0)
        return im == 0 and self.a <= re <= self.b

    def is_disjoint(self, other):
        """Return ``True`` if two isolation intervals are disjoint. """
        if isinstance(other, RealInterval):
            return self.b < other.a or other.b < self.a
        assert isinstance(other, ComplexInterval)
        return self.b < other.ax or other.bx < self.a or other.ay * other.by > 0

    def _inner_refine(self):
        """Internal one step real root refinement procedure. """
        if self.mobius is None:
            return self
        f, mobius = dup_inner_refine_real_root(self.f, self.mobius, self.dom, steps=1, mobius=True)
        return RealInterval(mobius + (self.neg,), f, self.dom)

    def refine_disjoint(self, other):
        """Refine an isolating interval until it is disjoint with another one. """
        expr = self
        while not expr.is_disjoint(other):
            expr, other = (expr._inner_refine(), other._inner_refine())
        return (expr, other)

    def refine_size(self, dx):
        """Refine an isolating interval until it is of sufficiently small size. """
        expr = self
        while not expr.dx < dx:
            expr = expr._inner_refine()
        return expr

    def refine_step(self, steps=1):
        """Perform several steps of real root refinement algorithm. """
        expr = self
        for _ in range(steps):
            expr = expr._inner_refine()
        return expr

    def refine(self):
        """Perform one step of real root refinement algorithm. """
        return self._inner_refine()