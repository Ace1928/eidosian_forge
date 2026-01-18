from sympy.core import Add, Mul, S
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I
from sympy.core.relational import Eq, Equality
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import (expand_mul, expand, Derivative,
from sympy.functions import (exp, im, cos, sin, re, Piecewise,
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices import zeros, Matrix, NonSquareMatrixError, MatrixBase, eye
from sympy.polys import Poly, together
from sympy.simplify import collect, radsimp, signsimp # type: ignore
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.sets.sets import FiniteSet
from sympy.solvers.deutils import ode_order
from sympy.solvers.solveset import NonlinearError, solveset
from sympy.utilities.iterables import (connected_components, iterable,
from sympy.utilities.misc import filldedent
from sympy.integrals.integrals import Integral, integrate
def _combine_type1_subsystems(subsystem, funcs, t):
    indices = [i for i, sys in enumerate(zip(subsystem, funcs)) if _is_type1(sys, t)]
    remove = set()
    for ip, i in enumerate(indices):
        for j in indices[ip + 1:]:
            if any((eq2.has(funcs[i]) for eq2 in subsystem[j])):
                subsystem[j] = subsystem[i] + subsystem[j]
                remove.add(i)
    subsystem = [sys for i, sys in enumerate(subsystem) if i not in remove]
    return subsystem