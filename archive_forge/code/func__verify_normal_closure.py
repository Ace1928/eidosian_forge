from sympy.combinatorics import Permutation
from sympy.combinatorics.util import _distribute_gens_by_base
def _verify_normal_closure(group, arg, closure=None):
    from sympy.combinatorics.perm_groups import PermutationGroup
    '\n    Verify the normal closure of a subgroup/subset/element in a group.\n\n    This is used to test\n    sympy.combinatorics.perm_groups.PermutationGroup.normal_closure\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.named_groups import (SymmetricGroup,\n    ... AlternatingGroup)\n    >>> from sympy.combinatorics.testutil import _verify_normal_closure\n    >>> S = SymmetricGroup(3)\n    >>> A = AlternatingGroup(3)\n    >>> _verify_normal_closure(S, A, closure=A)\n    True\n\n    See Also\n    ========\n\n    sympy.combinatorics.perm_groups.PermutationGroup.normal_closure\n\n    '
    if closure is None:
        closure = group.normal_closure(arg)
    conjugates = set()
    if hasattr(arg, 'generators'):
        subgr_gens = arg.generators
    elif hasattr(arg, '__getitem__'):
        subgr_gens = arg
    elif hasattr(arg, 'array_form'):
        subgr_gens = [arg]
    for el in group.generate_dimino():
        for gen in subgr_gens:
            conjugates.add(gen ^ el)
    naive_closure = PermutationGroup(list(conjugates))
    return closure.is_subgroup(naive_closure)