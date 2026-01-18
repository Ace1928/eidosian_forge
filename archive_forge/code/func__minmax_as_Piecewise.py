from sympy.core import Function, S, sympify, NumberKind
from sympy.utilities.iterables import sift
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.operations import LatticeOp, ShortCircuit
from sympy.core.function import (Application, Lambda,
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.relational import Eq, Relational
from sympy.core.singleton import Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.rules import Transform
from sympy.core.logic import fuzzy_and, fuzzy_or, _torf
from sympy.core.traversal import walk
from sympy.core.numbers import Integer
from sympy.logic.boolalg import And, Or
def _minmax_as_Piecewise(op, *args):
    from sympy.functions.elementary.piecewise import Piecewise
    ec = []
    for i, a in enumerate(args):
        c = [Relational(a, args[j], op) for j in range(i + 1, len(args))]
        ec.append((a, And(*c)))
    return Piecewise(*ec)