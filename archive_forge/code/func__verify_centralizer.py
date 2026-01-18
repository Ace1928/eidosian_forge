from sympy.combinatorics import Permutation
from sympy.combinatorics.util import _distribute_gens_by_base
def _verify_centralizer(group, arg, centr=None):
    """
    Verify the centralizer of a group/set/element inside another group.

    This is used for testing ``.centralizer()`` from
    ``sympy.combinatorics.perm_groups``

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
    ... AlternatingGroup)
    >>> from sympy.combinatorics.perm_groups import PermutationGroup
    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.combinatorics.testutil import _verify_centralizer
    >>> S = SymmetricGroup(5)
    >>> A = AlternatingGroup(5)
    >>> centr = PermutationGroup([Permutation([0, 1, 2, 3, 4])])
    >>> _verify_centralizer(S, A, centr)
    True

    See Also
    ========

    _naive_list_centralizer,
    sympy.combinatorics.perm_groups.PermutationGroup.centralizer,
    _cmp_perm_lists

    """
    if centr is None:
        centr = group.centralizer(arg)
    centr_list = list(centr.generate_dimino(af=True))
    centr_list_naive = _naive_list_centralizer(group, arg, af=True)
    return _cmp_perm_lists(centr_list, centr_list_naive)