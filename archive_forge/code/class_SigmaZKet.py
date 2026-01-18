from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.physics.quantum import Operator, Ket, Bra
from sympy.physics.quantum import ComplexSpace
from sympy.matrices import Matrix
from sympy.functions.special.tensor_functions import KroneckerDelta
class SigmaZKet(Ket):
    """Ket for a two-level system quantum system.

    Parameters
    ==========

    n : Number
        The state number (0 or 1).

    """

    def __new__(cls, n):
        if n not in (0, 1):
            raise ValueError('n must be 0 or 1')
        return Ket.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return SigmaZBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(2)

    def _eval_innerproduct_SigmaZBra(self, bra, **hints):
        return KroneckerDelta(self.n, bra.n)

    def _apply_from_right_to_SigmaZ(self, op, **options):
        if self.n == 0:
            return self
        else:
            return S.NegativeOne * self

    def _apply_from_right_to_SigmaX(self, op, **options):
        return SigmaZKet(1) if self.n == 0 else SigmaZKet(0)

    def _apply_from_right_to_SigmaY(self, op, **options):
        return I * SigmaZKet(1) if self.n == 0 else -I * SigmaZKet(0)

    def _apply_from_right_to_SigmaMinus(self, op, **options):
        if self.n == 0:
            return SigmaZKet(1)
        else:
            return S.Zero

    def _apply_from_right_to_SigmaPlus(self, op, **options):
        if self.n == 0:
            return S.Zero
        else:
            return SigmaZKet(0)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[1], [0]]) if self.n == 0 else Matrix([[0], [1]])
        else:
            raise NotImplementedError('Representation in format ' + format + ' not implemented.')