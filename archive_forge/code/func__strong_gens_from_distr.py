from sympy.combinatorics.permutations import Permutation, _af_invert, _af_rmul
from sympy.ntheory import isprime
def _strong_gens_from_distr(strong_gens_distr):
    """
    Retrieve strong generating set from generators of basic stabilizers.

    This is just the union of the generators of the first and second basic
    stabilizers.

    Parameters
    ==========

    ``strong_gens_distr`` - strong generators distributed by membership in basic
    stabilizers

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import (_strong_gens_from_distr,
    ... _distribute_gens_by_base)
    >>> S = SymmetricGroup(3)
    >>> S.schreier_sims()
    >>> S.strong_gens
    [(0 1 2), (2)(0 1), (1 2)]
    >>> strong_gens_distr = _distribute_gens_by_base(S.base, S.strong_gens)
    >>> _strong_gens_from_distr(strong_gens_distr)
    [(0 1 2), (2)(0 1), (1 2)]

    See Also
    ========

    _distribute_gens_by_base

    """
    if len(strong_gens_distr) == 1:
        return strong_gens_distr[0][:]
    else:
        result = strong_gens_distr[0]
        for gen in strong_gens_distr[1]:
            if gen not in result:
                result.append(gen)
        return result