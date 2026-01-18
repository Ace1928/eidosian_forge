from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
def _restrict_codomain(self, sm):
    """Implementation of codomain restriction."""
    return self.__class__(self.domain, sm, self.matrix)