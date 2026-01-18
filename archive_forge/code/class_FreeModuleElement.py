from copy import copy
from functools import reduce
from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable
class FreeModuleElement(ModuleElement):
    """Element of a free module. Data stored as a tuple."""

    def add(self, d1, d2):
        return tuple((x + y for x, y in zip(d1, d2)))

    def mul(self, d, p):
        return tuple((x * p for x in d))

    def div(self, d, p):
        return tuple((x / p for x in d))

    def __repr__(self):
        from sympy.printing.str import sstr
        return '[' + ', '.join((sstr(x) for x in self.data)) + ']'

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, idx):
        return self.data[idx]