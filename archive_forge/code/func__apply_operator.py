from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, expand)
from sympy.core.mul import Mul
from sympy.core.numbers import oo
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
from sympy.matrices import eye
def _apply_operator(self, ket, **options):
    return ket