from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
def _quotient(self, J, **opts):
    if not isinstance(J, ModuleImplementedIdeal):
        raise NotImplementedError
    return self._module.module_quotient(J._module, **opts)