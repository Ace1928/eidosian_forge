from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
def A4_in_S6():
    """
    Return a representation of A4 as a transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    G = PermutationGroup(Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 1, 2)(3, 5, 4))
    set_alternating_group_properties(G, 4, 6)
    return G