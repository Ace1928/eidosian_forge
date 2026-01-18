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
class SigmaX(SigmaOpBase):
    """Pauli sigma x operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent
    >>> from sympy.physics.quantum.pauli import SigmaX
    >>> sx = SigmaX()
    >>> sx
    SigmaX()
    >>> represent(sx)
    Matrix([
    [0, 1],
    [1, 0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args, **hints)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return 2 * I * SigmaZ(self.name)

    def _eval_commutator_SigmaZ(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return -2 * I * SigmaY(self.name)

    def _eval_commutator_BosonOp(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return S.Zero

    def _eval_adjoint(self):
        return self

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return '{\\sigma_x^{(%s)}}' % str(self.name)
        else:
            return '{\\sigma_x}'

    def _print_contents(self, printer, *args):
        return 'SigmaX()'

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return SigmaX(self.name).__pow__(int(e) % 2)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 1], [1, 0]])
        else:
            raise NotImplementedError('Representation in format ' + format + ' not implemented.')