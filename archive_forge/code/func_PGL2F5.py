from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
def PGL2F5():
    """
    Return a representation of the group $PGL_2(\\mathbb{F}_5)$, as a transitive
    subgroup of S6, isomorphic to $S_5$.

    Notes
    =====

    See [1], Page 325.

    """
    G = PermutationGroup(Permutation(0, 1, 2, 3, 4), Permutation(0, 5)(1, 2)(3, 4))
    set_symmetric_group_properties(G, 5, 6)
    return G