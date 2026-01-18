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
def _apply_operator_uncoupled(self, state, ket, *, dummy=True, **options):
    a = self.alpha
    b = self.beta
    g = self.gamma
    j = ket.j
    m = ket.m
    if j.is_number:
        s = []
        size = m_values(j)
        sz = size[1]
        for mp in sz:
            r = Rotation.D(j, m, mp, a, b, g)
            z = r.doit()
            s.append(z * state(j, mp))
        return Add(*s)
    else:
        if dummy:
            mp = Dummy('mp')
        else:
            mp = symbols('mp')
        return Sum(Rotation.D(j, m, mp, a, b, g) * state(j, mp), (mp, -j, j))