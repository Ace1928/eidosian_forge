from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
def _image(self):
    return self.codomain.submodule(*[self(x) for x in self.domain.gens])