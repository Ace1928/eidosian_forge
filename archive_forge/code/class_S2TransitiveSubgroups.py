from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
class S2TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S2.
    """
    S2 = 'S2'

    def get_perm_group(self):
        return SymmetricGroup(2)