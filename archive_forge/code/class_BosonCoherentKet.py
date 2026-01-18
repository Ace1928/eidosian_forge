from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, FockSpace, Ket, Bra, IdentityOperator
from sympy.functions.special.tensor_functions import KroneckerDelta
class BosonCoherentKet(Ket):
    """Coherent state ket for a bosonic mode.

    Parameters
    ==========

    alpha : Number, Symbol
        The complex amplitude of the coherent state.

    """

    def __new__(cls, alpha):
        return Ket.__new__(cls, alpha)

    @property
    def alpha(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonCoherentBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return HilbertSpace()

    def _eval_innerproduct_BosonCoherentBra(self, bra, **hints):
        if self.alpha == bra.alpha:
            return S.One
        else:
            return exp(-(abs(self.alpha) ** 2 + abs(bra.alpha) ** 2 - 2 * conjugate(bra.alpha) * self.alpha) / 2)

    def _apply_from_right_to_BosonOp(self, op, **options):
        if op.is_annihilation:
            return self.alpha * self
        else:
            return None