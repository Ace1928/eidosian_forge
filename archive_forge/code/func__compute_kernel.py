import itertools
from sympy.combinatorics.fp_groups import FpGroup, FpSubgroup, simplify_presentation
from sympy.combinatorics.free_groups import FreeGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.numbers import igcd
from sympy.ntheory.factor_ import totient
from sympy.core.singleton import S
def _compute_kernel(self):
    G = self.domain
    G_order = G.order()
    if G_order is S.Infinity:
        raise NotImplementedError('Kernel computation is not implemented for infinite groups')
    gens = []
    if isinstance(G, PermutationGroup):
        K = PermutationGroup(G.identity)
    else:
        K = FpSubgroup(G, gens, normal=True)
    i = self.image().order()
    while K.order() * i != G_order:
        r = G.random()
        k = r * self.invert(self(r)) ** (-1)
        if k not in K:
            gens.append(k)
            if isinstance(G, PermutationGroup):
                K = PermutationGroup(gens)
            else:
                K = FpSubgroup(G, gens, normal=True)
    return K