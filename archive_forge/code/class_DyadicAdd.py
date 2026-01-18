from __future__ import annotations
from sympy.vector.basisdependent import (BasisDependent, BasisDependentAdd,
from sympy.core import S, Pow
from sympy.core.expr import AtomicExpr
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
import sympy.vector
class DyadicAdd(BasisDependentAdd, Dyadic):
    """ Class to hold dyadic sums """

    def __new__(cls, *args, **options):
        obj = BasisDependentAdd.__new__(cls, *args, **options)
        return obj

    def _sympystr(self, printer):
        items = list(self.components.items())
        items.sort(key=lambda x: x[0].__str__())
        return ' + '.join((printer._print(k * v) for k, v in items))