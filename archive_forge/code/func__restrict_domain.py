from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
def _restrict_domain(self, sm):
    """Implementation of domain restriction."""
    return SubModuleHomomorphism(sm, self.codomain, self.matrix)