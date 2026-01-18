from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.polyerrors import (CoercionFailed, NotInvertible,
from sympy.polys.polytools import Poly
from sympy.printing.defaults import DefaultPrinting
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