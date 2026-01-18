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
class JminusOp(SpinOpBase, Operator):
    """The J- operator."""
    _coord = '-'
    basis = 'Jz'

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j
        m = ket.m
        if m.is_Number and j.is_Number:
            if m <= -j:
                return S.Zero
        return hbar * sqrt(j * (j + S.One) - m * (m - S.One)) * JzKet(j, m - S.One)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j
        m = ket.m
        jn = ket.jn
        coupling = ket.coupling
        if m.is_Number and j.is_Number:
            if m <= -j:
                return S.Zero
        return hbar * sqrt(j * (j + S.One) - m * (m - S.One)) * JzKetCoupled(j, m - S.One, jn, coupling)

    def matrix_element(self, j, m, jp, mp):
        result = hbar * sqrt(j * (j + S.One) - mp * (mp - S.One))
        result *= KroneckerDelta(m, mp - 1)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0]) - I * JyOp(args[0])