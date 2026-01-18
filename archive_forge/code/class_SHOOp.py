from sympy.core.numbers import (I, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.cartesian import X, Px
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.matrixutils import matrix_zeros
class SHOOp(Operator):
    """A base class for the SHO Operators.

    We are limiting the number of arguments to be 1.

    """

    @classmethod
    def _eval_args(cls, args):
        args = QExpr._eval_args(args)
        if len(args) == 1:
            return args
        else:
            raise ValueError('Too many arguments')

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(S.Infinity)