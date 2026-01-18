from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
def G36p():
    """
    Return a representation of the group G36+, a transitive subgroup of S6
    isomorphic to the semidirect product of C3^2 with C4.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    return PermutationGroup(Permutation(5)(0, 1, 2), Permutation(3, 4, 5), Permutation(0, 5, 2, 3)(1, 4))