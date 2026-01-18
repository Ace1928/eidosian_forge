from sympy.polys.polytools import Poly
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
from sympy.utilities.decorator import public
from .basis import round_two, nilradical_mod_p
from .exceptions import StructureError
from .modules import ModuleEndomorphism, find_min_poly
from .utilities import coeff_search, supplement_a_subspace
def _compute_test_factor(p, gens, ZK):
    """
    Compute the test factor for a :py:class:`~.PrimeIdeal` $\\mathfrak{p}$.

    Parameters
    ==========

    p : int
        The rational prime $\\mathfrak{p}$ divides

    gens : list of :py:class:`PowerBasisElement`
        A complete set of generators for $\\mathfrak{p}$ over *ZK*, EXCEPT that
        an element equivalent to rational *p* can and should be omitted (since
        it has no effect except to waste time).

    ZK : :py:class:`~.Submodule`
        The maximal order where the prime ideal $\\mathfrak{p}$ lives.

    Returns
    =======

    :py:class:`~.PowerBasisElement`

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
    (See Proposition 4.8.15.)

    """
    _check_formal_conditions_for_maximal_order(ZK)
    E = ZK.endomorphism_ring()
    matrices = [E.inner_endomorphism(g).matrix(modulus=p) for g in gens]
    B = DomainMatrix.zeros((0, ZK.n), FF(p)).vstack(*matrices)
    x = B.nullspace()[0, :].transpose()
    beta = ZK.parent(ZK.matrix * x, denom=ZK.denom)
    return beta