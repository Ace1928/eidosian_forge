from sympy.polys.domains.integerring import ZZ
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.normalforms import (
def _to_domain(m, domain=None):
    """Convert Matrix to DomainMatrix"""
    ring = getattr(m, 'ring', None)
    m = m.applyfunc(lambda e: e.as_expr() if isinstance(e, Poly) else e)
    dM = DomainMatrix.from_Matrix(m)
    domain = domain or ring
    if domain is not None:
        dM = dM.convert_to(domain)
    return dM