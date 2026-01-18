from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def _trace_D(gj, p_i, Dxtrav):
    """
    Return the representative h satisfying h[gj] == p_i

    If there is not such a representative return None
    """
    for h in Dxtrav:
        if h[gj] == p_i:
            return h
    return None