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
class SpinOpBase:
    """Base class for spin operators."""

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(S.Infinity)

    @property
    def name(self):
        return self.args[0]

    def _print_contents(self, printer, *args):
        return '%s%s' % (self.name, self._coord)

    def _print_contents_pretty(self, printer, *args):
        a = stringPict(str(self.name))
        b = stringPict(self._coord)
        return self._print_subscript_pretty(a, b)

    def _print_contents_latex(self, printer, *args):
        return '%s_%s' % (self.name, self._coord)

    def _represent_base(self, basis, **options):
        j = options.get('j', S.Half)
        size, mvals = m_values(j)
        result = zeros(size, size)
        for p in range(size):
            for q in range(size):
                me = self.matrix_element(j, mvals[p], j, mvals[q])
                result[p, q] = me
        return result

    def _apply_op(self, ket, orig_basis, **options):
        state = ket.rewrite(self.basis)
        if isinstance(state, State):
            ret = hbar * state.m * state
        elif isinstance(state, Sum):
            ret = self._apply_operator_Sum(state, **options)
        else:
            ret = qapply(self * state)
        if ret == self * state:
            raise NotImplementedError
        return ret.rewrite(orig_basis)

    def _apply_operator_JxKet(self, ket, **options):
        return self._apply_op(ket, 'Jx', **options)

    def _apply_operator_JxKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jx', **options)

    def _apply_operator_JyKet(self, ket, **options):
        return self._apply_op(ket, 'Jy', **options)

    def _apply_operator_JyKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jy', **options)

    def _apply_operator_JzKet(self, ket, **options):
        return self._apply_op(ket, 'Jz', **options)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jz', **options)

    def _apply_operator_TensorProduct(self, tp, **options):
        if not isinstance(self, (JxOp, JyOp, JzOp)):
            raise NotImplementedError
        result = []
        for n in range(len(tp.args)):
            arg = []
            arg.extend(tp.args[:n])
            arg.append(self._apply_operator(tp.args[n]))
            arg.extend(tp.args[n + 1:])
            result.append(tp.__class__(*arg))
        return Add(*result).expand()

    def _apply_operator_Sum(self, s, **options):
        new_func = qapply(self * s.function)
        if new_func == self * s.function:
            raise NotImplementedError
        return Sum(new_func, *s.limits)

    def _eval_trace(self, **options):
        return self._represent_default_basis().trace()