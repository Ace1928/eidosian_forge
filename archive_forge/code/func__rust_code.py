from sympy.core import (S, pi, oo, symbols, Rational, Integer,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.logic import ITE
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import MatrixSymbol, SparseMatrix, Matrix
from sympy.printing.rust import rust_code
def _rust_code(self, printer):
    return '%s.fabs()' % printer._print(self.args[0])