from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sympify import sympify
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.matrixutils import (
from sympy.physics.quantum.trace import Tr
def _eval_rewrite(self, rule, args, **hints):
    return TensorProduct(*args).expand(tensorproduct=True)