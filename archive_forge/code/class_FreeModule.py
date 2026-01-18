from copy import copy
from functools import reduce
from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable
class FreeModule(Module):
    """
    Abstract base class for free modules.

    Additional attributes:

    - rank - rank of the free module

    Non-implemented methods:

    - submodule
    """
    dtype = FreeModuleElement

    def __init__(self, ring, rank):
        Module.__init__(self, ring)
        self.rank = rank

    def __repr__(self):
        return repr(self.ring) + '**' + repr(self.rank)

    def is_submodule(self, other):
        """
        Returns True if ``other`` is a submodule of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([2, x])
        >>> F.is_submodule(F)
        True
        >>> F.is_submodule(M)
        True
        >>> M.is_submodule(F)
        False
        """
        if isinstance(other, SubModule):
            return other.container == self
        if isinstance(other, FreeModule):
            return other.ring == self.ring and other.rank == self.rank
        return False

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal representation.

        This method is called implicitly whenever computations involve elements
        not in the internal representation.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.convert([1, 0])
        [1, 0]
        """
        if isinstance(elem, FreeModuleElement):
            if elem.module is self:
                return elem
            if elem.module.rank != self.rank:
                raise CoercionFailed
            return FreeModuleElement(self, tuple((self.ring.convert(x, elem.module.ring) for x in elem.data)))
        elif iterable(elem):
            tpl = tuple((self.ring.convert(x) for x in elem))
            if len(tpl) != self.rank:
                raise CoercionFailed
            return FreeModuleElement(self, tpl)
        elif _aresame(elem, 0):
            return FreeModuleElement(self, (self.ring.convert(0),) * self.rank)
        else:
            raise CoercionFailed

    def is_zero(self):
        """
        Returns True if ``self`` is a zero module.

        (If, as this implementation assumes, the coefficient ring is not the
        zero ring, then this is equivalent to the rank being zero.)

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(0).is_zero()
        True
        >>> QQ.old_poly_ring(x).free_module(1).is_zero()
        False
        """
        return self.rank == 0

    def basis(self):
        """
        Return a set of basis elements.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(3).basis()
        ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        """
        from sympy.matrices import eye
        M = eye(self.rank)
        return tuple((self.convert(M.row(i)) for i in range(self.rank)))

    def quotient_module(self, submodule):
        """
        Return a quotient module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2)
        >>> M.quotient_module(M.submodule([1, x], [x, 2]))
        QQ[x]**2/<[1, x], [x, 2]>

        Or more conicisely, using the overloaded division operator:

        >>> QQ.old_poly_ring(x).free_module(2) / [[1, x], [x, 2]]
        QQ[x]**2/<[1, x], [x, 2]>
        """
        return QuotientModule(self.ring, self, submodule)

    def multiply_ideal(self, other):
        """
        Multiply ``self`` by the ideal ``other``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x)
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.multiply_ideal(I)
        <[x, 0], [0, x]>
        """
        return self.submodule(*self.basis()).multiply_ideal(other)

    def identity_hom(self):
        """
        Return the identity homomorphism on ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).identity_hom()
        Matrix([
        [1, 0], : QQ[x]**2 -> QQ[x]**2
        [0, 1]])
        """
        from sympy.polys.agca.homomorphisms import homomorphism
        return homomorphism(self, self, self.basis())