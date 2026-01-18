from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, FockSpace, Ket, Bra, IdentityOperator
from sympy.functions.special.tensor_functions import KroneckerDelta
class BosonCoherentBra(Bra):
    """Coherent state bra for a bosonic mode.

    Parameters
    ==========

    alpha : Number, Symbol
        The complex amplitude of the coherent state.

    """

    def __new__(cls, alpha):
        return Bra.__new__(cls, alpha)

    @property
    def alpha(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return BosonCoherentKet

    def _apply_operator_BosonOp(self, op, **options):
        if not op.is_annihilation:
            return self.alpha * self
        else:
            return None