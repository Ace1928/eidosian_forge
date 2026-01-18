from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
def _contains_ideal(self, J):
    if not isinstance(J, ModuleImplementedIdeal):
        raise NotImplementedError
    return self._module.is_submodule(J._module)