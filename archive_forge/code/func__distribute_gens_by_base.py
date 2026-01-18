from sympy.combinatorics.permutations import Permutation, _af_invert, _af_rmul
from sympy.ntheory import isprime
def _distribute_gens_by_base(base, gens):
    """
    Distribute the group elements ``gens`` by membership in basic stabilizers.

    Explanation
    ===========

    Notice that for a base `(b_1, b_2, \\dots, b_k)`, the basic stabilizers
    are defined as `G^{(i)} = G_{b_1, \\dots, b_{i-1}}` for
    `i \\in\\{1, 2, \\dots, k\\}`.

    Parameters
    ==========

    base : a sequence of points in `\\{0, 1, \\dots, n-1\\}`
    gens : a list of elements of a permutation group of degree `n`.

    Returns
    =======

    List of length `k`, where `k` is
    the length of ``base``. The `i`-th entry contains those elements in
    ``gens`` which fix the first `i` elements of ``base`` (so that the
    `0`-th entry is equal to ``gens`` itself). If no element fixes the first
    `i` elements of ``base``, the `i`-th element is set to a list containing
    the identity element.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> from sympy.combinatorics.util import _distribute_gens_by_base
    >>> D = DihedralGroup(3)
    >>> D.schreier_sims()
    >>> D.strong_gens
    [(0 1 2), (0 2), (1 2)]
    >>> D.base
    [0, 1]
    >>> _distribute_gens_by_base(D.base, D.strong_gens)
    [[(0 1 2), (0 2), (1 2)],
     [(1 2)]]

    See Also
    ========

    _strong_gens_from_distr, _orbits_transversals_from_bsgs,
    _handle_precomputed_bsgs

    """
    base_len = len(base)
    degree = gens[0].size
    stabs = [[] for _ in range(base_len)]
    max_stab_index = 0
    for gen in gens:
        j = 0
        while j < base_len - 1 and gen._array_form[base[j]] == base[j]:
            j += 1
        if j > max_stab_index:
            max_stab_index = j
        for k in range(j + 1):
            stabs[k].append(gen)
    for i in range(max_stab_index + 1, base_len):
        stabs[i].append(_af_new(list(range(degree))))
    return stabs