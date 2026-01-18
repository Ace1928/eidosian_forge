from copy import copy
from functools import reduce
from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable
def _module_quotient(self, other, relations=False):
    if relations and len(other.gens) != 1:
        raise NotImplementedError
    if len(other.gens) == 0:
        return self.ring.ideal(1)
    elif len(other.gens) == 1:
        g1 = list(other.gens[0]) + [1]
        gi = [list(x) + [0] for x in self.gens]
        M = self.ring.free_module(self.rank + 1).submodule(*[g1] + gi, order='ilex', TOP=False)
        if not relations:
            return self.ring.ideal(*[x[-1] for x in M._groebner_vec() if all((y == self.ring.zero for y in x[:-1]))])
        else:
            G, R = M._groebner_vec(extended=True)
            indices = [i for i, x in enumerate(G) if all((y == self.ring.zero for y in x[:-1]))]
            return (self.ring.ideal(*[G[i][-1] for i in indices]), [[-x for x in R[i][1:]] for i in indices])
    return reduce(lambda x, y: x.intersect(y), (self._module_quotient(self.container.submodule(x)) for x in other.gens))