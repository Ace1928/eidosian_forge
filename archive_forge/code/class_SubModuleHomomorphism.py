from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
class SubModuleHomomorphism(MatrixHomomorphism):
    """
    Concrete class for homomorphism with domain a submodule of a free module
    or a quotient thereof.

    Do not instantiate; the constructor does not check that your data is well
    defined. Use the ``homomorphism`` function instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> M = QQ.old_poly_ring(x).free_module(2)*x
    >>> homomorphism(M, M, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : <[x, 0], [0, x]> -> <[x, 0], [0, x]>
    [0, 1]])
    """

    def _apply(self, elem):
        if isinstance(self.domain, SubQuotientModule):
            elem = elem.data
        return sum((x * e for x, e in zip(elem, self.matrix)))

    def _image(self):
        return self.codomain.submodule(*[self(x) for x in self.domain.gens])

    def _kernel(self):
        syz = self.image().syzygy_module()
        return self.domain.submodule(*[sum((xi * gi for xi, gi in zip(s, self.domain.gens))) for s in syz.gens])