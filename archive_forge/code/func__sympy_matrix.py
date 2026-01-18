from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
def _sympy_matrix(self):
    """Helper function which returns a SymPy matrix ``self.matrix``."""
    from sympy.matrices import Matrix
    c = lambda x: x
    if isinstance(self.codomain, (QuotientModule, SubQuotientModule)):
        c = lambda x: x.data
    return Matrix([[self.ring.to_sympy(y) for y in c(x)] for x in self.matrix]).T