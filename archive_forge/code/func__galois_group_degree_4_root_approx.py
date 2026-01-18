from collections import defaultdict
import random
from sympy.core.symbol import Dummy, symbols
from sympy.ntheory.primetest import is_square
from sympy.polys.domains import ZZ
from sympy.polys.densebasic import dup_random
from sympy.polys.densetools import dup_eval
from sympy.polys.euclidtools import dup_discriminant
from sympy.polys.factortools import dup_factor_list, dup_irreducible_p
from sympy.polys.numberfields.galois_resolvents import (
from sympy.polys.numberfields.utilities import coeff_search
from sympy.polys.polytools import (Poly, poly_from_expr,
from sympy.polys.sqfreetools import dup_sqf_p
from sympy.utilities import public
def _galois_group_degree_4_root_approx(T, max_tries=30, randomize=False):
    """
    Compute the Galois group of a polynomial of degree 4.

    Explanation
    ===========

    Follows Alg 6.3.7 of [1], using a pure root approximation approach.

    """
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.galois import S4TransitiveSubgroups
    X = symbols('X0 X1 X2 X3')
    F1 = X[0] * X[2] + X[1] * X[3]
    s1 = [Permutation(3), Permutation(3)(0, 1), Permutation(3)(0, 3)]
    R1 = Resolvent(F1, X, s1)
    F2_pre = X[0] * X[1] ** 2 + X[1] * X[2] ** 2 + X[2] * X[3] ** 2 + X[3] * X[0] ** 2
    s2_pre = [Permutation(3), Permutation(3)(0, 2)]
    history = set()
    for i in range(max_tries):
        if i > 0:
            _, T = tschirnhausen_transformation(T, max_tries=max_tries, history=history, fixed_order=not randomize)
        R_dup, _, i0 = R1.eval_for_poly(T, find_integer_root=True)
        if not dup_sqf_p(R_dup, ZZ):
            continue
        sq_disc = has_square_disc(T)
        if i0 is None:
            return (S4TransitiveSubgroups.A4, True) if sq_disc else (S4TransitiveSubgroups.S4, False)
        if sq_disc:
            return (S4TransitiveSubgroups.V, True)
        sigma = s1[i0]
        F2 = F2_pre.subs(zip(X, sigma(X)), simultaneous=True)
        s2 = [sigma * tau * sigma for tau in s2_pre]
        R2 = Resolvent(F2, X, s2)
        R_dup, _, _ = R2.eval_for_poly(T)
        d = dup_discriminant(R_dup, ZZ)
        if d == 0:
            continue
        if is_square(d):
            return (S4TransitiveSubgroups.C4, False)
        else:
            return (S4TransitiveSubgroups.D4, False)
    raise MaxTriesException