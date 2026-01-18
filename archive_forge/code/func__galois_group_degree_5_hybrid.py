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
def _galois_group_degree_5_hybrid(T, max_tries=30, randomize=False):
    """
    Compute the Galois group of a polynomial of degree 5.

    Explanation
    ===========

    Based on Alg 6.3.9 of [1], but uses a hybrid approach, combining resolvent
    coeff lookup, with root approximation.

    """
    from sympy.combinatorics.galois import S5TransitiveSubgroups
    from sympy.combinatorics.permutations import Permutation
    X5 = symbols('X0,X1,X2,X3,X4')
    res = define_resolvents()
    F51, _, s51 = res[5, 1]
    F51 = F51.as_expr(*X5)
    R51 = Resolvent(F51, X5, s51)
    history = set()
    reached_second_stage = False
    for i in range(max_tries):
        if i > 0:
            _, T = tschirnhausen_transformation(T, max_tries=max_tries, history=history, fixed_order=not randomize)
        R51_dup = get_resolvent_by_lookup(T, 1)
        if not dup_sqf_p(R51_dup, ZZ):
            continue
        if not reached_second_stage:
            sq_disc = has_square_disc(T)
            if dup_irreducible_p(R51_dup, ZZ):
                return (S5TransitiveSubgroups.A5, True) if sq_disc else (S5TransitiveSubgroups.S5, False)
            if not sq_disc:
                return (S5TransitiveSubgroups.M20, False)
        reached_second_stage = True
        rounded_roots = R51.round_roots_to_integers_for_poly(T)
        for permutation_index, candidate_root in rounded_roots.items():
            if not dup_eval(R51_dup, candidate_root, ZZ):
                break
        X = X5
        F2_pre = X[0] * X[1] ** 2 + X[1] * X[2] ** 2 + X[2] * X[3] ** 2 + X[3] * X[4] ** 2 + X[4] * X[0] ** 2
        s2_pre = [Permutation(4), Permutation(4)(0, 1)(2, 4)]
        i0 = permutation_index
        sigma = s51[i0]
        F2 = F2_pre.subs(zip(X, sigma(X)), simultaneous=True)
        s2 = [sigma * tau * sigma for tau in s2_pre]
        R2 = Resolvent(F2, X, s2)
        R_dup, _, _ = R2.eval_for_poly(T)
        d = dup_discriminant(R_dup, ZZ)
        if d == 0:
            continue
        if is_square(d):
            return (S5TransitiveSubgroups.C5, True)
        else:
            return (S5TransitiveSubgroups.D5, True)
    raise MaxTriesException