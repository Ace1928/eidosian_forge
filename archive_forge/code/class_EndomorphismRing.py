from sympy.core.numbers import igcd, ilcm
from sympy.core.symbol import Dummy
from sympy.polys.polyclasses import ANP
from sympy.polys.polytools import Poly
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMBadInputError
from sympy.polys.matrices.normalforms import hermite_normal_form
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed
from sympy.polys.polyutils import IntegerPowerable
from .exceptions import ClosureFailure, MissingUnityError, StructureError
from .utilities import AlgIntPowers, is_rat, get_num_denom
class EndomorphismRing:
    """The ring of endomorphisms on a module."""

    def __init__(self, domain):
        """
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The domain and codomain of the endomorphisms.

        """
        self.domain = domain

    def inner_endomorphism(self, multiplier):
        """
        Form an inner endomorphism belonging to this endomorphism ring.

        Parameters
        ==========

        multiplier : :py:class:`~.ModuleElement`
            Element $a$ defining the inner endomorphism $x \\mapsto a x$.

        Returns
        =======

        :py:class:`~.InnerEndomorphism`

        """
        return InnerEndomorphism(self.domain, multiplier)

    def represent(self, element):
        """
        Represent an element of this endomorphism ring, as a single column
        vector.

        Explanation
        ===========

        Let $M$ be a module, and $E$ its ring of endomorphisms. Let $N$ be
        another module, and consider a homomorphism $\\varphi: N \\rightarrow E$.
        In the event that $\\varphi$ is to be represented by a matrix $A$, each
        column of $A$ must represent an element of $E$. This is possible when
        the elements of $E$ are themselves representable as matrices, by
        stacking the columns of such a matrix into a single column.

        This method supports calculating such matrices $A$, by representing
        an element of this endomorphism ring first as a matrix, and then
        stacking that matrix's columns into a single column.

        Examples
        ========

        Note that in these examples we print matrix transposes, to make their
        columns easier to inspect.

        >>> from sympy import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> from sympy.polys.numberfields.modules import ModuleHomomorphism
        >>> T = Poly(cyclotomic_poly(5))
        >>> M = PowerBasis(T)
        >>> E = M.endomorphism_ring()

        Let $\\zeta$ be a primitive 5th root of unity, a generator of our field,
        and consider the inner endomorphism $\\tau$ on the ring of integers,
        induced by $\\zeta$:

        >>> zeta = M(1)
        >>> tau = E.inner_endomorphism(zeta)
        >>> tau.matrix().transpose()  # doctest: +SKIP
        DomainMatrix(
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]],
            (4, 4), ZZ)

        The matrix representation of $\\tau$ is as expected. The first column
        shows that multiplying by $\\zeta$ carries $1$ to $\\zeta$, the second
        column that it carries $\\zeta$ to $\\zeta^2$, and so forth.

        The ``represent`` method of the endomorphism ring ``E`` stacks these
        into a single column:

        >>> E.represent(tau).transpose()  # doctest: +SKIP
        DomainMatrix(
            [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, -1, -1, -1]],
            (1, 16), ZZ)

        This is useful when we want to consider a homomorphism $\\varphi$ having
        ``E`` as codomain:

        >>> phi = ModuleHomomorphism(M, E, lambda x: E.inner_endomorphism(x))

        and we want to compute the matrix of such a homomorphism:

        >>> phi.matrix().transpose()  # doctest: +SKIP
        DomainMatrix(
            [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, -1, -1, -1],
            [0, 0, 1, 0, 0, 0, 0, 1, -1, -1, -1, -1, 1, 0, 0, 0],
            [0, 0, 0, 1, -1, -1, -1, -1, 1, 0, 0, 0, 0, 1, 0, 0]],
            (4, 16), ZZ)

        Note that the stacked matrix of $\\tau$ occurs as the second column in
        this example. This is because $\\zeta$ is the second basis element of
        ``M``, and $\\varphi(\\zeta) = \\tau$.

        Parameters
        ==========

        element : :py:class:`~.ModuleEndomorphism` belonging to this ring.

        Returns
        =======

        :py:class:`~.DomainMatrix`
            Column vector equalling the vertical stacking of all the columns
            of the matrix that represents the given *element* as a mapping.

        """
        if isinstance(element, ModuleEndomorphism) and element.domain == self.domain:
            M = element.matrix()
            m, n = M.shape
            if n == 0:
                return M
            return M[:, 0].vstack(*[M[:, j] for j in range(1, n)])
        raise NotImplementedError