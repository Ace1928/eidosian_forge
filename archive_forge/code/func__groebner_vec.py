from copy import copy
from functools import reduce
from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable
def _groebner_vec(self, extended=False):
    """Returns a standard basis in element form."""
    if not extended:
        return [FreeModuleElement(self, tuple(self.ring._sdm_to_vector(x, self.rank))) for x in self._groebner()]
    gb, gbe = self._groebner(extended=True)
    return ([self.convert(self.ring._sdm_to_vector(x, self.rank)) for x in gb], [self.ring._sdm_to_vector(x, len(self.gens)) for x in gbe])