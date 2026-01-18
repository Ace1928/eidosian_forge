from the names used in Bronstein's book.
from types import GeneratorType
from functools import reduce
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh,
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, cos,
from .integrals import integrate, Integral
from .heurisch import _symbols
from sympy.polys.polyerrors import DomainError, PolynomialError
from sympy.polys.polytools import (real_roots, cancel, Poly, gcd,
from sympy.polys.rootoftools import RootSum
from sympy.utilities.iterables import numbered_symbols
def _rewrite_logs(self, logs, symlogs):
    """
        Rewrite logs for better processing.
        """
    atoms = self.newf.atoms(log)
    logs = update_sets(logs, atoms, lambda i: i.args[0].is_rational_function(*self.T) and i.args[0].has(*self.T))
    symlogs = update_sets(symlogs, atoms, lambda i: i.has(*self.T) and i.args[0].is_Pow and i.args[0].base.is_rational_function(*self.T) and (not i.args[0].exp.is_Integer))
    for i in ordered(symlogs):
        lbase = log(i.args[0].base)
        logs.append(lbase)
        new = i.args[0].exp * lbase
        self.newf = self.newf.xreplace({i: new})
        self.backsubs.append((new, i))
    logs = sorted(set(logs), key=default_sort_key)
    return (logs, symlogs)