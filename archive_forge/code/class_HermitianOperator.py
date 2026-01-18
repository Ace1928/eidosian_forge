from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, expand)
from sympy.core.mul import Mul
from sympy.core.numbers import oo
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
from sympy.matrices import eye
class HermitianOperator(Operator):
    """A Hermitian operator that satisfies H == Dagger(H).

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, HermitianOperator
    >>> H = HermitianOperator('H')
    >>> Dagger(H)
    H
    """
    is_hermitian = True

    def _eval_inverse(self):
        if isinstance(self, UnitaryOperator):
            return self
        else:
            return Operator._eval_inverse(self)

    def _eval_power(self, exp):
        if isinstance(self, UnitaryOperator):
            if exp.is_even:
                from sympy.core.singleton import S
                return S.One
            elif exp.is_odd:
                return self
        return Operator._eval_power(self, exp)