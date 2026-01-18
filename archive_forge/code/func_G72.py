from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
def G72():
    """
    Return a representation of the group G72, a transitive subgroup of S6
    isomorphic to the semidirect product of C3^2 with D4.

    Notes
    =====

    See [1], Page 325.

    """
    return PermutationGroup(Permutation(5)(0, 1, 2), Permutation(0, 4, 1, 3)(2, 5), Permutation(0, 3)(1, 4)(2, 5))