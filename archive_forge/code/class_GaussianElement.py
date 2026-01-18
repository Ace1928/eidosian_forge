from sympy.core.numbers import I
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.ring import Ring
class GaussianElement(DomainElement):
    """Base class for elements of Gaussian type domains."""
    base: Domain
    _parent: Domain
    __slots__ = ('x', 'y')

    def __new__(cls, x, y=0):
        conv = cls.base.convert
        return cls.new(conv(x), conv(y))

    @classmethod
    def new(cls, x, y):
        """Create a new GaussianElement of the same domain."""
        obj = super().__new__(cls)
        obj.x = x
        obj.y = y
        return obj

    def parent(self):
        """The domain that this is an element of (ZZ_I or QQ_I)"""
        return self._parent

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y
        else:
            return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, GaussianElement):
            return NotImplemented
        return [self.y, self.x] < [other.y, other.x]

    def __pos__(self):
        return self

    def __neg__(self):
        return self.new(-self.x, -self.y)

    def __repr__(self):
        return '%s(%s, %s)' % (self._parent.rep, self.x, self.y)

    def __str__(self):
        return str(self._parent.to_sympy(self))

    @classmethod
    def _get_xy(cls, other):
        if not isinstance(other, cls):
            try:
                other = cls._parent.convert(other)
            except CoercionFailed:
                return (None, None)
        return (other.x, other.y)

    def __add__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(self.x + x, self.y + y)
        else:
            return NotImplemented
    __radd__ = __add__

    def __sub__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(self.x - x, self.y - y)
        else:
            return NotImplemented

    def __rsub__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(x - self.x, y - self.y)
        else:
            return NotImplemented

    def __mul__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(self.x * x - self.y * y, self.x * y + self.y * x)
        else:
            return NotImplemented
    __rmul__ = __mul__

    def __pow__(self, exp):
        if exp == 0:
            return self.new(1, 0)
        if exp < 0:
            self, exp = (1 / self, -exp)
        if exp == 1:
            return self
        pow2 = self
        prod = self if exp % 2 else self._parent.one
        exp //= 2
        while exp:
            pow2 *= pow2
            if exp % 2:
                prod *= pow2
            exp //= 2
        return prod

    def __bool__(self):
        return bool(self.x) or bool(self.y)

    def quadrant(self):
        """Return quadrant index 0-3.

        0 is included in quadrant 0.
        """
        if self.y > 0:
            return 0 if self.x > 0 else 1
        elif self.y < 0:
            return 2 if self.x < 0 else 3
        else:
            return 0 if self.x >= 0 else 2

    def __rdivmod__(self, other):
        try:
            other = self._parent.convert(other)
        except CoercionFailed:
            return NotImplemented
        else:
            return other.__divmod__(self)

    def __rtruediv__(self, other):
        try:
            other = QQ_I.convert(other)
        except CoercionFailed:
            return NotImplemented
        else:
            return other.__truediv__(self)

    def __floordiv__(self, other):
        qr = self.__divmod__(other)
        return qr if qr is NotImplemented else qr[0]

    def __rfloordiv__(self, other):
        qr = self.__rdivmod__(other)
        return qr if qr is NotImplemented else qr[0]

    def __mod__(self, other):
        qr = self.__divmod__(other)
        return qr if qr is NotImplemented else qr[1]

    def __rmod__(self, other):
        qr = self.__rdivmod__(other)
        return qr if qr is NotImplemented else qr[1]