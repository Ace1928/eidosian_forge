from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.simplify import simplify
from sympy.matrices import zeros
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import pretty_symbol
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.operator import (HermitianOperator, Operator,
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import ComplexSpace, DirectSumHilbertSpace
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.qapply import qapply
class SpinState(State):
    """Base class for angular momentum states."""
    _label_separator = ','

    def __new__(cls, j, m):
        j = sympify(j)
        m = sympify(m)
        if j.is_number:
            if 2 * j != int(2 * j):
                raise ValueError('j must be integer or half-integer, got: %s' % j)
            if j < 0:
                raise ValueError('j must be >= 0, got: %s' % j)
        if m.is_number:
            if 2 * m != int(2 * m):
                raise ValueError('m must be integer or half-integer, got: %s' % m)
        if j.is_number and m.is_number:
            if abs(m) > j:
                raise ValueError('Allowed values for m are -j <= m <= j, got j, m: %s, %s' % (j, m))
            if int(j - m) != j - m:
                raise ValueError('Both j and m must be integer or half-integer, got j, m: %s, %s' % (j, m))
        return State.__new__(cls, j, m)

    @property
    def j(self):
        return self.label[0]

    @property
    def m(self):
        return self.label[1]

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(2 * label[0] + 1)

    def _represent_base(self, **options):
        j = self.j
        m = self.m
        alpha = sympify(options.get('alpha', 0))
        beta = sympify(options.get('beta', 0))
        gamma = sympify(options.get('gamma', 0))
        size, mvals = m_values(j)
        result = zeros(size, 1)
        for p, mval in enumerate(mvals):
            if m.is_number:
                result[p, 0] = Rotation.D(self.j, mval, self.m, alpha, beta, gamma).doit()
            else:
                result[p, 0] = Rotation.D(self.j, mval, self.m, alpha, beta, gamma)
        return result

    def _eval_rewrite_as_Jx(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jx, JxBra, **options)
        return self._rewrite_basis(Jx, JxKet, **options)

    def _eval_rewrite_as_Jy(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jy, JyBra, **options)
        return self._rewrite_basis(Jy, JyKet, **options)

    def _eval_rewrite_as_Jz(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jz, JzBra, **options)
        return self._rewrite_basis(Jz, JzKet, **options)

    def _rewrite_basis(self, basis, evect, **options):
        from sympy.physics.quantum.represent import represent
        j = self.j
        args = self.args[2:]
        if j.is_number:
            if isinstance(self, CoupledSpinState):
                if j == int(j):
                    start = j ** 2
                else:
                    start = (2 * j - 1) * (2 * j + 1) / 4
            else:
                start = 0
            vect = represent(self, basis=basis, **options)
            result = Add(*[vect[start + i] * evect(j, j - i, *args) for i in range(2 * j + 1)])
            if isinstance(self, CoupledSpinState) and options.get('coupled') is False:
                return uncouple(result)
            return result
        else:
            i = 0
            mi = symbols('mi')
            while self.subs(mi, 0) != self:
                i += 1
                mi = symbols('mi%d' % i)
                break
            if isinstance(self, CoupledSpinState):
                test_args = (0, mi, (0, 0))
            else:
                test_args = (0, mi)
            if isinstance(self, Ket):
                angles = represent(self.__class__(*test_args), basis=basis)[0].args[3:6]
            else:
                angles = represent(self.__class__(*test_args), basis=basis)[0].args[0].args[3:6]
            if angles == (0, 0, 0):
                return self
            else:
                state = evect(j, mi, *args)
                lt = Rotation.D(j, mi, self.m, *angles)
                return Sum(lt * state, (mi, -j, j))

    def _eval_innerproduct_JxBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JxOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(self.j, bra.j) * KroneckerDelta(self.m, bra.m)
        return result

    def _eval_innerproduct_JyBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JyOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(self.j, bra.j) * KroneckerDelta(self.m, bra.m)
        return result

    def _eval_innerproduct_JzBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JzOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(self.j, bra.j) * KroneckerDelta(self.m, bra.m)
        return result

    def _eval_trace(self, bra, **hints):
        return (bra * self).doit()