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
def _build_coupled(jcoupling, length):
    n_list = [[n + 1] for n in range(length)]
    coupled_jn = []
    coupled_n = []
    for n1, n2, j_new in jcoupling:
        coupled_jn.append(j_new)
        coupled_n.append((n_list[n1 - 1], n_list[n2 - 1]))
        n_sort = sorted(n_list[n1 - 1] + n_list[n2 - 1])
        n_list[n_sort[0] - 1] = n_sort
    return (coupled_n, coupled_jn)