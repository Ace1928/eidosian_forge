from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
def _quotient_domain(self, sm):
    """Implementation of domain quotient."""
    return self.__class__(self.domain / sm, self.codomain, self.matrix)