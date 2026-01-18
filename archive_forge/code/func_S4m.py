from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
def S4m():
    """
    Return a representation of the S4- transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    G = PermutationGroup(Permutation(1, 4, 5, 3), Permutation(0, 4)(1, 5)(2, 3))
    set_symmetric_group_properties(G, 4, 6)
    return G