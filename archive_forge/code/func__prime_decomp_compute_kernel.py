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
def _prime_decomp_compute_kernel(I, p, ZK):
    """
    Parameters
    ==========

    I : :py:class:`~.Module`
        An ideal of ``ZK/pZK``.
    p : int
        The rational prime being factored.
    ZK : :py:class:`~.Submodule`
        The maximal order.

    Returns
    =======

    Pair ``(N, G)``, where:

        ``N`` is a :py:class:`~.Module` representing the kernel of the map
        ``a |--> a**p - a`` on ``(O/pO)/I``, guaranteed to be a module with
        unity.

        ``G`` is a :py:class:`~.Module` representing a basis for the separable
        algebra ``A = O/I`` (see Cohen).

    """
    W = I.matrix
    n, r = W.shape
    if r == 0:
        B = W.eye(n, ZZ)
    else:
        B = W.hstack(W.eye(n, ZZ)[:, 0])
    if B.shape[1] < n:
        B = supplement_a_subspace(B.convert_to(FF(p))).convert_to(ZZ)
    G = ZK.submodule_from_matrix(B)
    G.compute_mult_tab()
    G = G.discard_before(r)
    phi = ModuleEndomorphism(G, lambda x: x ** p - x)
    N = phi.kernel(modulus=p)
    assert N.starts_with_unity()
    return (N, G)