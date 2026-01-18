from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.polyerrors import (CoercionFailed, NotInvertible,
from sympy.polys.polytools import Poly
from sympy.printing.defaults import DefaultPrinting
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