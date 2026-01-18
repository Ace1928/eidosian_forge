from sympy.polys.polytools import Poly
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.utilities.decorator import public
from .modules import ModuleEndomorphism, ModuleHomomorphism, PowerBasis
from .utilities import extract_fundamental_discriminant
def _second_enlargement(H, p, q):
    """
    Perform the second enlargement in the Round Two algorithm.
    """
    Ip = nilradical_mod_p(H, p, q=q)
    B = H.parent.submodule_from_matrix(H.matrix * Ip.matrix, denom=H.denom)
    C = B + p * H
    E = C.endomorphism_ring()
    phi = ModuleHomomorphism(H, E, lambda x: E.inner_endomorphism(x))
    gamma = phi.kernel(modulus=p)
    G = H.parent.submodule_from_matrix(H.matrix * gamma.matrix, denom=H.denom * p)
    H1 = G + H
    return (H1, Ip)