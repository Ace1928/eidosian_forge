from sympy.core.numbers import I
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.ring import Ring
@classmethod
def _get_xy(cls, other):
    if not isinstance(other, cls):
        try:
            other = cls._parent.convert(other)
        except CoercionFailed:
            return (None, None)
    return (other.x, other.y)