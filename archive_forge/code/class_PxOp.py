from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.delta_functions import DiracDelta
from sympy.sets.sets import Interval
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import L2
from sympy.physics.quantum.operator import DifferentialOperator, HermitianOperator
from sympy.physics.quantum.state import Ket, Bra, State
class PxOp(HermitianOperator):
    """1D cartesian momentum operator."""

    @classmethod
    def default_args(self):
        return ('Px',)

    @classmethod
    def _eval_hilbert_space(self, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _apply_operator_PxKet(self, ket, **options):
        return ket.momentum * ket

    def _represent_XKet(self, basis, *, index=1, **options):
        states = basis._enumerate_state(2, start_index=index)
        coord1 = states[0].position
        coord2 = states[1].position
        d = DifferentialOperator(coord1)
        delta = DiracDelta(coord1 - coord2)
        return -I * hbar * (d * delta)