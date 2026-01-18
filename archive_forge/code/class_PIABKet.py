from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.sets.sets import Interval
from sympy.physics.quantum.operator import HermitianOperator
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.constants import hbar
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import L2
class PIABKet(Ket):
    """Particle in a box eigenket."""

    @classmethod
    def _eval_hilbert_space(cls, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    @classmethod
    def dual_class(self):
        return PIABBra

    def _represent_default_basis(self, **options):
        return self._represent_XOp(None, **options)

    def _represent_XOp(self, basis, **options):
        x = Symbol('x')
        n = Symbol('n')
        subs_info = options.get('subs', {})
        return sqrt(2 / L) * sin(n * pi * x / L).subs(subs_info)

    def _eval_innerproduct_PIABBra(self, bra):
        return KroneckerDelta(bra.label[0], self.label[0])