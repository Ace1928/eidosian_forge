from sympy.combinatorics.group_constructs import DirectProduct
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
def AbelianGroup(*cyclic_orders):
    """
    Returns the direct product of cyclic groups with the given orders.

    Explanation
    ===========

    According to the structure theorem for finite abelian groups ([1]),
    every finite abelian group can be written as the direct product of
    finitely many cyclic groups.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import AbelianGroup
    >>> AbelianGroup(3, 4)
    PermutationGroup([
            (6)(0 1 2),
            (3 4 5 6)])
    >>> _.is_group
    True

    See Also
    ========

    DirectProduct

    References
    ==========

    .. [1] https://groupprops.subwiki.org/wiki/Structure_theorem_for_finitely_generated_abelian_groups

    """
    groups = []
    degree = 0
    order = 1
    for size in cyclic_orders:
        degree += size
        order *= size
        groups.append(CyclicGroup(size))
    G = DirectProduct(*groups)
    G._is_abelian = True
    G._degree = degree
    G._order = order
    return G