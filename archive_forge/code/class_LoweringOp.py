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
class LoweringOp(SHOOp):
    """The Lowering Operator or 'a'.

    When 'a' acts on a state it lowers the state up by one. Taking
    the adjoint of 'a' returns a^dagger, the Raising Operator. 'a'
    can be rewritten in terms of position and momentum. We can
    represent 'a' as a matrix, which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Lowering Operator and rewrite it in terms of position and
    momentum, and show that taking its adjoint returns a^dagger:

        >>> from sympy.physics.quantum.sho1d import LoweringOp
        >>> from sympy.physics.quantum import Dagger

        >>> a = LoweringOp('a')
        >>> a.rewrite('xp').doit()
        sqrt(2)*(m*omega*X + I*Px)/(2*sqrt(hbar)*sqrt(m*omega))

        >>> Dagger(a)
        RaisingOp(a)

    Taking the commutator of 'a' with other Operators:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import LoweringOp, RaisingOp
        >>> from sympy.physics.quantum.sho1d import NumberOp

        >>> a = LoweringOp('a')
        >>> ad = RaisingOp('a')
        >>> N = NumberOp('N')
        >>> Commutator(a, ad).doit()
        1
        >>> Commutator(a, N).doit()
        a

    Apply 'a' to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import LoweringOp, SHOKet

        >>> a = LoweringOp('a')
        >>> k = SHOKet('k')
        >>> qapply(a*k)
        sqrt(k)*|k - 1>

    Taking 'a' of the lowest state will return 0:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import LoweringOp, SHOKet

        >>> a = LoweringOp('a')
        >>> k = SHOKet(0)
        >>> qapply(a*k)
        0

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import LoweringOp
        >>> from sympy.physics.quantum.represent import represent
        >>> a = LoweringOp('a')
        >>> represent(a, basis=N, ndim=4, format='sympy')
        Matrix([
        [0, 1,       0,       0],
        [0, 0, sqrt(2),       0],
        [0, 0,       0, sqrt(3)],
        [0, 0,       0,       0]])

    """

    def _eval_rewrite_as_xp(self, *args, **kwargs):
        return S.One / sqrt(Integer(2) * hbar * m * omega) * (I * Px + m * omega * X)

    def _eval_adjoint(self):
        return RaisingOp(*self.args)

    def _eval_commutator_RaisingOp(self, other):
        return S.One

    def _eval_commutator_NumberOp(self, other):
        return self

    def _apply_operator_SHOKet(self, ket, **options):
        temp = ket.n - Integer(1)
        if ket.n is S.Zero:
            return S.Zero
        else:
            return sqrt(ket.n) * SHOKet(temp)

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        raise NotImplementedError('Position representation is not implemented')

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        for i in range(ndim_info - 1):
            value = sqrt(i + 1)
            if format == 'scipy.sparse':
                value = float(value)
            matrix[i, i + 1] = value
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        return matrix