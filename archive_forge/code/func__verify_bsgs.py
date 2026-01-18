from sympy.combinatorics import Permutation
from sympy.combinatorics.util import _distribute_gens_by_base
def _verify_bsgs(group, base, gens):
    """
    Verify the correctness of a base and strong generating set.

    Explanation
    ===========

    This is a naive implementation using the definition of a base and a strong
    generating set relative to it. There are other procedures for
    verifying a base and strong generating set, but this one will
    serve for more robust testing.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import AlternatingGroup
    >>> from sympy.combinatorics.testutil import _verify_bsgs
    >>> A = AlternatingGroup(4)
    >>> A.schreier_sims()
    >>> _verify_bsgs(A, A.base, A.strong_gens)
    True

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.schreier_sims

    """
    from sympy.combinatorics.perm_groups import PermutationGroup
    strong_gens_distr = _distribute_gens_by_base(base, gens)
    current_stabilizer = group
    for i in range(len(base)):
        candidate = PermutationGroup(strong_gens_distr[i])
        if current_stabilizer.order() != candidate.order():
            return False
        current_stabilizer = current_stabilizer.stabilizer(base[i])
    if current_stabilizer.order() != 1:
        return False
    return True