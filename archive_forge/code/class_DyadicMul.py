from __future__ import annotations
from sympy.vector.basisdependent import (BasisDependent, BasisDependentAdd,
from sympy.core import S, Pow
from sympy.core.expr import AtomicExpr
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
import sympy.vector
class DyadicMul(BasisDependentMul, Dyadic):
    """ Products of scalars and BaseDyadics """

    def __new__(cls, *args, **options):
        obj = BasisDependentMul.__new__(cls, *args, **options)
        return obj

    @property
    def base_dyadic(self):
        """ The BaseDyadic involved in the product. """
        return self._base_instance

    @property
    def measure_number(self):
        """ The scalar expression involved in the definition of
        this DyadicMul.
        """
        return self._measure_number