from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
class FreeModuleHomomorphism(MatrixHomomorphism):
    """
    Concrete class for homomorphisms with domain a free module or a quotient
    thereof.

    Do not instantiate; the constructor does not check that your data is well
    defined. Use the ``homomorphism`` function instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> homomorphism(F, F, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : QQ[x]**2 -> QQ[x]**2
    [0, 1]])
    """

    def _apply(self, elem):
        if isinstance(self.domain, QuotientModule):
            elem = elem.data
        return sum((x * e for x, e in zip(elem, self.matrix)))

    def _image(self):
        return self.codomain.submodule(*self.matrix)

    def _kernel(self):
        syz = self.image().syzygy_module()
        return self.domain.submodule(*syz.gens)