from sympy.core.add import Add
from sympy.core.exprtools import factor_terms
from sympy.core.function import expand_log, _mexpand
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.miscellaneous import root
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, factor
from sympy.simplify.simplify import separatevars
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import powsimp
from sympy.solvers.solvers import solve, _invert
from sympy.utilities.iterables import uniq
def _linab(arg, symbol):
    """Return ``a, b, X`` assuming ``arg`` can be written as ``a*X + b``
    where ``X`` is a symbol-dependent factor and ``a`` and ``b`` are
    independent of ``symbol``.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _linab
    >>> from sympy.abc import x, y
    >>> from sympy import exp, S
    >>> _linab(S(2), x)
    (2, 0, 1)
    >>> _linab(2*x, x)
    (2, 0, x)
    >>> _linab(y + y*x + 2*x, x)
    (y + 2, y, x)
    >>> _linab(3 + 2*exp(x), x)
    (2, 3, exp(x))
    """
    arg = factor_terms(arg.expand())
    ind, dep = arg.as_independent(symbol)
    if arg.is_Mul and dep.is_Add:
        a, b, x = _linab(dep, symbol)
        return (ind * a, ind * b, x)
    if not arg.is_Add:
        b = 0
        a, x = (ind, dep)
    else:
        b = ind
        a, x = separatevars(dep).as_independent(symbol, as_Add=False)
    if x.could_extract_minus_sign():
        a = -a
        x = -x
    return (a, b, x)