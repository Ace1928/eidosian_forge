from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def _is_minimal_bsgs(base, gens):
    """
    Check if the BSGS has minimal base under lexigographic order.

    base, gens BSGS

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import riemann_bsgs, _is_minimal_bsgs
    >>> _is_minimal_bsgs(*riemann_bsgs)
    True
    >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))
    >>> _is_minimal_bsgs(*riemann_bsgs1)
    False
    """
    base1 = []
    sgs1 = gens[:]
    size = gens[0].size
    for i in range(size):
        if not all((h._array_form[i] == i for h in sgs1)):
            base1.append(i)
            sgs1 = [h for h in sgs1 if h._array_form[i] == i]
    return base1 == base