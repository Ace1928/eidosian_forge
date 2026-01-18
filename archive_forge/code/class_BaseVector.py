from __future__ import annotations
from itertools import product
from sympy.core.add import Add
from sympy.core.assumptions import StdFactKB
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.vector.basisdependent import (BasisDependentZero,
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.dyadic import Dyadic, BaseDyadic, DyadicAdd
class BaseVector(Vector, AtomicExpr):
    """
    Class to denote a base vector.

    """

    def __new__(cls, index, system, pretty_str=None, latex_str=None):
        if pretty_str is None:
            pretty_str = 'x{}'.format(index)
        if latex_str is None:
            latex_str = 'x_{}'.format(index)
        pretty_str = str(pretty_str)
        latex_str = str(latex_str)
        if index not in range(0, 3):
            raise ValueError('index must be 0, 1 or 2')
        if not isinstance(system, CoordSys3D):
            raise TypeError('system should be a CoordSys3D')
        name = system._vector_names[index]
        obj = super().__new__(cls, S(index), system)
        obj._base_instance = obj
        obj._components = {obj: S.One}
        obj._measure_number = S.One
        obj._name = system._name + '.' + name
        obj._pretty_form = '' + pretty_str
        obj._latex_form = latex_str
        obj._system = system
        obj._id = (index, system)
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        obj._sys = system
        return obj

    @property
    def system(self):
        return self._system

    def _sympystr(self, printer):
        return self._name

    def _sympyrepr(self, printer):
        index, system = self._id
        return printer._print(system) + '.' + system._vector_names[index]

    @property
    def free_symbols(self):
        return {self}