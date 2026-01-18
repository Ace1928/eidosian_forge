from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.mul import Mul
from sympy.core.relational import Equality, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import (piecewise_fold,
from sympy.logic.boolalg import BooleanFunction
from sympy.matrices.matrices import MatrixBase
from sympy.sets.sets import Interval, Set
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx
from sympy.utilities import flatten
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.exceptions import sympy_deprecation_warning
def _common_new(cls, function, *symbols, discrete, **assumptions):
    """Return either a special return value or the tuple,
    (function, limits, orientation). This code is common to
    both ExprWithLimits and AddWithLimits."""
    function = sympify(function)
    if isinstance(function, Equality):
        limits, orientation = _process_limits(*symbols, discrete=discrete)
        if not (limits and all((len(limit) == 3 for limit in limits))):
            sympy_deprecation_warning('\n                Creating a indefinite integral with an Eq() argument is\n                deprecated.\n\n                This is because indefinite integrals do not preserve equality\n                due to the arbitrary constants. If you want an equality of\n                indefinite integrals, use Eq(Integral(a, x), Integral(b, x))\n                explicitly.\n                ', deprecated_since_version='1.6', active_deprecations_target='deprecated-indefinite-integral-eq', stacklevel=5)
        lhs = function.lhs
        rhs = function.rhs
        return Equality(cls(lhs, *symbols, **assumptions), cls(rhs, *symbols, **assumptions))
    if function is S.NaN:
        return S.NaN
    if symbols:
        limits, orientation = _process_limits(*symbols, discrete=discrete)
        for i, li in enumerate(limits):
            if len(li) == 4:
                function = function.subs(li[0], li[-1])
                limits[i] = Tuple(*li[:-1])
    else:
        free = function.free_symbols
        if len(free) != 1:
            raise ValueError('specify dummy variables for %s' % function)
        limits, orientation = ([Tuple(s) for s in free], 1)
    while cls == type(function):
        limits = list(function.limits) + limits
        function = function.function
    reps = {}
    symbols_of_integration = {i[0] for i in limits}
    for p in function.atoms(Piecewise):
        if not p.has(*symbols_of_integration):
            reps[p] = Dummy()
    function = function.xreplace(reps)
    function = piecewise_fold(function)
    function = function.xreplace({v: k for k, v in reps.items()})
    return (function, limits, orientation)