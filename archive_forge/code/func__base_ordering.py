from sympy.combinatorics.permutations import Permutation, _af_invert, _af_rmul
from sympy.ntheory import isprime
def _base_ordering(base, degree):
    """
    Order `\\{0, 1, \\dots, n-1\\}` so that base points come first and in order.

    Parameters
    ==========

    base : the base
    degree : the degree of the associated permutation group

    Returns
    =======

    A list ``base_ordering`` such that ``base_ordering[point]`` is the
    number of ``point`` in the ordering.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _base_ordering
    >>> S = SymmetricGroup(4)
    >>> S.schreier_sims()
    >>> _base_ordering(S.base, S.degree)
    [0, 1, 2, 3]

    Notes
    =====

    This is used in backtrack searches, when we define a relation `\\ll` on
    the underlying set for a permutation group of degree `n`,
    `\\{0, 1, \\dots, n-1\\}`, so that if `(b_1, b_2, \\dots, b_k)` is a base we
    have `b_i \\ll b_j` whenever `i<j` and `b_i \\ll a` for all
    `i\\in\\{1,2, \\dots, k\\}` and `a` is not in the base. The idea is developed
    and applied to backtracking algorithms in [1], pp.108-132. The points
    that are not in the base are taken in increasing order.

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of computational group theory"

    """
    base_len = len(base)
    ordering = [0] * degree
    for i in range(base_len):
        ordering[base[i]] = i
    current = base_len
    for i in range(degree):
        if i not in base:
            ordering[i] = current
            current += 1
    return ordering