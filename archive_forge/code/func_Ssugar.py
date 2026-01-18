from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def Ssugar(i, j):
    """Compute the sugar of the S-poly corresponding to (i, j)."""
    LMi = sdm_LM(S[i])
    LMj = sdm_LM(S[j])
    return max(Sugars[i] - sdm_monomial_deg(LMi), Sugars[j] - sdm_monomial_deg(LMj)) + sdm_monomial_deg(sdm_monomial_lcm(LMi, LMj))