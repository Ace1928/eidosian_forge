from sympy.polys.agca.modules import FreeModuleQuotientRing
from sympy.polys.domains.ring import Ring
from sympy.polys.polyerrors import NotReversible, CoercionFailed
from sympy.utilities import public
@public
class QuotientRingElement:
    """
    Class representing elements of (commutative) quotient rings.

    Attributes:

    - ring - containing ring
    - data - element of ring.ring (i.e. base ring) representing self
    """

    def __init__(self, ring, data):
        self.ring = ring
        self.data = data

    def __str__(self):
        from sympy.printing.str import sstr
        return sstr(self.data) + ' + ' + str(self.ring.base_ideal)
    __repr__ = __str__

    def __bool__(self):
        return not self.ring.is_zero(self)

    def __add__(self, om):
        if not isinstance(om, self.__class__) or om.ring != self.ring:
            try:
                om = self.ring.convert(om)
            except (NotImplementedError, CoercionFailed):
                return NotImplemented
        return self.ring(self.data + om.data)
    __radd__ = __add__

    def __neg__(self):
        return self.ring(self.data * self.ring.ring.convert(-1))

    def __sub__(self, om):
        return self.__add__(-om)

    def __rsub__(self, om):
        return (-self).__add__(om)

    def __mul__(self, o):
        if not isinstance(o, self.__class__):
            try:
                o = self.ring.convert(o)
            except (NotImplementedError, CoercionFailed):
                return NotImplemented
        return self.ring(self.data * o.data)
    __rmul__ = __mul__

    def __rtruediv__(self, o):
        return self.ring.revert(self) * o

    def __truediv__(self, o):
        if not isinstance(o, self.__class__):
            try:
                o = self.ring.convert(o)
            except (NotImplementedError, CoercionFailed):
                return NotImplemented
        return self.ring.revert(o) * self

    def __pow__(self, oth):
        if oth < 0:
            return self.ring.revert(self) ** (-oth)
        return self.ring(self.data ** oth)

    def __eq__(self, om):
        if not isinstance(om, self.__class__) or om.ring != self.ring:
            return False
        return self.ring.is_zero(self - om)

    def __ne__(self, om):
        return not self == om