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
def _galois_group_degree_3(T, max_tries=30, randomize=False):
    """
    Compute the Galois group of a polynomial of degree 3.

    Explanation
    ===========

    Uses Prop 6.3.5 of [1].

    """
    from sympy.combinatorics.galois import S3TransitiveSubgroups
    return (S3TransitiveSubgroups.A3, True) if has_square_disc(T) else (S3TransitiveSubgroups.S3, False)