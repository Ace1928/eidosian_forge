from sympy.combinatorics.permutations import Permutation, _af_invert, _af_rmul
from sympy.ntheory import isprime
def _orbits_transversals_from_bsgs(base, strong_gens_distr, transversals_only=False, slp=False):
    """
    Compute basic orbits and transversals from a base and strong generating set.

    Explanation
    ===========

    The generators are provided as distributed across the basic stabilizers.
    If the optional argument ``transversals_only`` is set to True, only the
    transversals are returned.

    Parameters
    ==========

    ``base`` - The base.
    ``strong_gens_distr`` - Strong generators distributed by membership in basic
    stabilizers.
    ``transversals_only`` - bool
        A flag switching between returning only the
        transversals and both orbits and transversals.
    ``slp`` -
        If ``True``, return a list of dictionaries containing the
        generator presentations of the elements of the transversals,
        i.e. the list of indices of generators from ``strong_gens_distr[i]``
        such that their product is the relevant transversal element.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _distribute_gens_by_base
    >>> S = SymmetricGroup(3)
    >>> S.schreier_sims()
    >>> strong_gens_distr = _distribute_gens_by_base(S.base, S.strong_gens)
    >>> (S.base, strong_gens_distr)
    ([0, 1], [[(0 1 2), (2)(0 1), (1 2)], [(1 2)]])

    See Also
    ========

    _distribute_gens_by_base, _handle_precomputed_bsgs

    """
    from sympy.combinatorics.perm_groups import _orbit_transversal
    base_len = len(base)
    degree = strong_gens_distr[0][0].size
    transversals = [None] * base_len
    slps = [None] * base_len
    if transversals_only is False:
        basic_orbits = [None] * base_len
    for i in range(base_len):
        transversals[i], slps[i] = _orbit_transversal(degree, strong_gens_distr[i], base[i], pairs=True, slp=True)
        transversals[i] = dict(transversals[i])
        if transversals_only is False:
            basic_orbits[i] = list(transversals[i].keys())
    if transversals_only:
        return transversals
    else:
        if not slp:
            return (basic_orbits, transversals)
        return (basic_orbits, transversals, slps)