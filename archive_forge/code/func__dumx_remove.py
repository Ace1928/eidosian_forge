from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def _dumx_remove(dumx, dumx_flat, p0):
    """
    remove p0 from dumx
    """
    res = []
    for dx in dumx:
        if p0 not in dx:
            res.append(dx)
            continue
        k = dx.index(p0)
        if k % 2 == 0:
            p0_paired = dx[k + 1]
        else:
            p0_paired = dx[k - 1]
        dx.remove(p0)
        dx.remove(p0_paired)
        dumx_flat.remove(p0)
        dumx_flat.remove(p0_paired)
        res.append(dx)