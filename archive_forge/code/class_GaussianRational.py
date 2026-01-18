from sympy.core.numbers import I
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.ring import Ring
class GaussianRational(GaussianElement):
    """Gaussian rational: domain element for :ref:`QQ_I`

        >>> from sympy import QQ_I, QQ
        >>> z = QQ_I(QQ(2, 3), QQ(4, 5))
        >>> z
        (2/3 + 4/5*I)
        >>> type(z)
        <class 'sympy.polys.domains.gaussiandomains.GaussianRational'>
    """
    base = QQ

    def __truediv__(self, other):
        """Return a Gaussian rational."""
        if not other:
            raise ZeroDivisionError('{} / 0'.format(self))
        x, y = self._get_xy(other)
        if x is None:
            return NotImplemented
        c = x * x + y * y
        return GaussianRational((self.x * x + self.y * y) / c, (-self.x * y + self.y * x) / c)

    def __divmod__(self, other):
        try:
            other = self._parent.convert(other)
        except CoercionFailed:
            return NotImplemented
        if not other:
            raise ZeroDivisionError('{} % 0'.format(self))
        else:
            return (self / other, QQ_I.zero)