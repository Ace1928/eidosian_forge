from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
class S5TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S5.
    """
    C5 = 'C5'
    D5 = 'D5'
    M20 = 'M20'
    A5 = 'A5'
    S5 = 'S5'

    def get_perm_group(self):
        if self == S5TransitiveSubgroups.C5:
            return CyclicGroup(5)
        elif self == S5TransitiveSubgroups.D5:
            return DihedralGroup(5)
        elif self == S5TransitiveSubgroups.M20:
            return M20()
        elif self == S5TransitiveSubgroups.A5:
            return AlternatingGroup(5)
        elif self == S5TransitiveSubgroups.S5:
            return SymmetricGroup(5)