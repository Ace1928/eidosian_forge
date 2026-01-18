from sympy.core.numbers import I
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.ring import Ring
class GaussianInteger(GaussianElement):
    """Gaussian integer: domain element for :ref:`ZZ_I`

        >>> from sympy import ZZ_I
        >>> z = ZZ_I(2, 3)
        >>> z
        (2 + 3*I)
        >>> type(z)
        <class 'sympy.polys.domains.gaussiandomains.GaussianInteger'>
    """
    base = ZZ

    def __truediv__(self, other):
        """Return a Gaussian rational."""
        return QQ_I.convert(self) / other

    def __divmod__(self, other):
        if not other:
            raise ZeroDivisionError('divmod({}, 0)'.format(self))
        x, y = self._get_xy(other)
        if x is None:
            return NotImplemented
        a, b = (self.x * x + self.y * y, -self.x * y + self.y * x)
        c = x * x + y * y
        qx = (2 * a + c) // (2 * c)
        qy = (2 * b + c) // (2 * c)
        q = GaussianInteger(qx, qy)
        return (q, self - q * other)