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
def _to_perm_group(self):
    """
        Return an isomorphic permutation group and the isomorphism.
        The implementation is dependent on coset enumeration so
        will only terminate for finite groups.

        """
    from sympy.combinatorics import Permutation
    from sympy.combinatorics.homomorphisms import homomorphism
    if self.order() is S.Infinity:
        raise NotImplementedError('Permutation presentation of infinite groups is not implemented')
    if self._perm_isomorphism:
        T = self._perm_isomorphism
        P = T.image()
    else:
        C = self.coset_table([])
        gens = self.generators
        images = [[C[i][2 * gens.index(g)] for i in range(len(C))] for g in gens]
        images = [Permutation(i) for i in images]
        P = PermutationGroup(images)
        T = homomorphism(self, P, gens, images, check=False)
        self._perm_isomorphism = T
    return (P, T)