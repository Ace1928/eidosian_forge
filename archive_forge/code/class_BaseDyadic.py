from __future__ import annotations
from sympy.vector.basisdependent import (BasisDependent, BasisDependentAdd,
from sympy.core import S, Pow
from sympy.core.expr import AtomicExpr
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
import sympy.vector
class BaseDyadic(Dyadic, AtomicExpr):
    """
    Class to denote a base dyadic tensor component.
    """

    def __new__(cls, vector1, vector2):
        Vector = sympy.vector.Vector
        BaseVector = sympy.vector.BaseVector
        VectorZero = sympy.vector.VectorZero
        if not isinstance(vector1, (BaseVector, VectorZero)) or not isinstance(vector2, (BaseVector, VectorZero)):
            raise TypeError('BaseDyadic cannot be composed of non-base ' + 'vectors')
        elif vector1 == Vector.zero or vector2 == Vector.zero:
            return Dyadic.zero
        obj = super().__new__(cls, vector1, vector2)
        obj._base_instance = obj
        obj._measure_number = 1
        obj._components = {obj: S.One}
        obj._sys = vector1._sys
        obj._pretty_form = '(' + vector1._pretty_form + '|' + vector2._pretty_form + ')'
        obj._latex_form = '\\left(' + vector1._latex_form + '{\\middle|}' + vector2._latex_form + '\\right)'
        return obj

    def _sympystr(self, printer):
        return '({}|{})'.format(printer._print(self.args[0]), printer._print(self.args[1]))

    def _sympyrepr(self, printer):
        return 'BaseDyadic({}, {})'.format(printer._print(self.args[0]), printer._print(self.args[1]))