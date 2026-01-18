from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def _get_bsgs(G, base, gens, free_indices):
    """
        return the BSGS for G.pointwise_stabilizer(free_indices)
        """
    if not free_indices:
        return (base[:], gens[:])
    else:
        H = G.pointwise_stabilizer(free_indices)
        base, sgs = H.schreier_sims_incremental()
        return (base, sgs)