from copy import copy
from functools import reduce
from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable
class ModuleOrder(ProductOrder):
    """A product monomial order with a zeroth term as module index."""

    def __init__(self, o1, o2, TOP):
        if TOP:
            ProductOrder.__init__(self, (o2, _subs1), (o1, _subs0))
        else:
            ProductOrder.__init__(self, (o1, _subs0), (o2, _subs1))