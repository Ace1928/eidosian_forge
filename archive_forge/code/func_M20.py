from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
def M20():
    """
    Return a representation of the metacyclic group M20, a transitive subgroup
    of S5 that is one of the possible Galois groups for polys of degree 5.

    Notes
    =====

    See [1], Page 323.

    """
    G = PermutationGroup(Permutation(0, 1, 2, 3, 4), Permutation(1, 2, 4, 3))
    G._degree = 5
    G._order = 20
    G._is_transitive = True
    G._is_sym = False
    G._is_alt = False
    G._is_cyclic = False
    G._is_dihedral = False
    return G