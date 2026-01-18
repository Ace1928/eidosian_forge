from copy import copy
from functools import reduce
from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable
class SubQuotientModule(SubModule):
    """
    Submodule of a quotient module.

    Equivalently, quotient module of a submodule.

    Do not instantiate this, instead use the submodule or quotient_module
    constructing methods:

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> S = F.submodule([1, 0], [1, x])
    >>> Q = F/[(1, 0)]
    >>> S/[(1, 0)] == Q.submodule([5, x])
    True

    Attributes:

    - base - base module we are quotient of
    - killed_module - submodule used to form the quotient
    """

    def __init__(self, gens, container, **opts):
        SubModule.__init__(self, gens, container)
        self.killed_module = self.container.killed_module
        self.base = self.container.base.submodule(*[x.data for x in self.gens], **opts).union(self.killed_module)

    def _contains(self, elem):
        return self.base.contains(elem.data)

    def _syzygies(self):
        return [X[:len(self.gens)] for X in self.base._syzygies()]

    def _in_terms_of_generators(self, e):
        return self.base._in_terms_of_generators(e.data)[:len(self.gens)]

    def is_full_module(self):
        """
        Return True if ``self`` is the entire free module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_full_module()
        False
        >>> F.submodule([1, 1], [1, 2]).is_full_module()
        True
        """
        return self.base.is_full_module()

    def quotient_hom(self):
        """
        Return the quotient homomorphism to self.

        That is, return the natural map from ``self.base`` to ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = (QQ.old_poly_ring(x).free_module(2) / [(1, x)]).submodule([1, 0])
        >>> M.quotient_hom()
        Matrix([
        [1, 0], : <[1, 0], [1, x]> -> <[1, 0] + <[1, x]>, [1, x] + <[1, x]>>
        [0, 1]])
        """
        return self.base.identity_hom().quotient_codomain(self.killed_module)