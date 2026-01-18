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
def _prime_decomp_split_ideal(I, p, N, G, ZK):
    """
    Perform the step in the prime decomposition algorithm where we have determined
    the the quotient ``ZK/I`` is _not_ a field, and we want to perform a non-trivial
    factorization of *I* by locating an idempotent element of ``ZK/I``.
    """
    assert I.parent == ZK and G.parent is ZK and (N.parent is G)
    alpha = N(1).to_parent()
    assert alpha.module is G
    alpha_powers = []
    m = find_min_poly(alpha, FF(p), powers=alpha_powers)
    lc, fl = m.factor_list()
    m1 = fl[0][0]
    m2 = m.quo(m1)
    U, V, g = m1.gcdex(m2)
    assert g == 1
    E = list(reversed(Poly(U * m1, domain=ZZ).rep.rep))
    eps1 = sum((E[i] * alpha_powers[i] for i in range(len(E))))
    eps2 = 1 - eps1
    idemps = [eps1, eps2]
    factors = []
    for eps in idemps:
        e = eps.to_parent()
        assert e.module is ZK
        D = I.matrix.convert_to(FF(p)).hstack(*[(e * om).column(domain=FF(p)) for om in ZK.basis_elements()])
        W = D.columnspace().convert_to(ZZ)
        H = ZK.submodule_from_matrix(W)
        factors.append(H)
    return factors