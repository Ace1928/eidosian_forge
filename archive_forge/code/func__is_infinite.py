from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.combinatorics.free_groups import (FreeGroup, FreeGroupElement,
from sympy.combinatorics.rewritingsystem import RewritingSystem
from sympy.combinatorics.coset_table import (CosetTable,
from sympy.combinatorics import PermutationGroup
from sympy.matrices.normalforms import invariant_factors
from sympy.matrices import Matrix
from sympy.polys.polytools import gcd
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.magic import pollute
from itertools import product
def _is_infinite(self):
    """
        Test if the group is infinite. Return `True` if the test succeeds
        and `None` otherwise

        """
    used_gens = set()
    for r in self.relators:
        used_gens.update(r.contains_generators())
    if not set(self.generators) <= used_gens:
        return True
    abelian_rels = []
    for rel in self.relators:
        abelian_rels.append([rel.exponent_sum(g) for g in self.generators])
    m = Matrix(Matrix(abelian_rels))
    if 0 in invariant_factors(m):
        return True
    else:
        return None