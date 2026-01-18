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
def _log_part(self, logs):
    """
        Try to build a logarithmic extension.

        Returns
        =======

        Returns True if there was a new extension and False if there was no new
        extension but it was able to rewrite the given logarithms in terms
        of the existing extension.  Unlike with exponential extensions, there
        is no way that a logarithm is not transcendental over and cannot be
        rewritten in terms of an already existing extension in a non-algebraic
        way, so this function does not ever return None or raise
        NotImplementedError.
        """
    from .prde import is_deriv_k
    new_extension = False
    logargs = [i.args[0] for i in logs]
    for arg in ordered(logargs):
        arga, argd = frac_in(arg, self.t)
        A = is_deriv_k(arga, argd, self)
        if A is not None:
            ans, u, const = A
            newterm = log(const) + u
            self.newf = self.newf.xreplace({log(arg): newterm})
            continue
        else:
            arga, argd = frac_in(arg, self.t)
            darga = argd * derivation(Poly(arga, self.t), self) - arga * derivation(Poly(argd, self.t), self)
            dargd = argd ** 2
            darg = darga.as_expr() / dargd.as_expr()
            self.t = next(self.ts)
            self.T.append(self.t)
            self.extargs.append(arg)
            self.exts.append('log')
            self.D.append(cancel(darg.as_expr() / arg).as_poly(self.t, expand=False))
            if self.dummy:
                i = Dummy('i')
            else:
                i = Symbol('i')
            self.Tfuncs += [Lambda(i, log(arg.subs(self.x, i)))]
            self.newf = self.newf.xreplace({log(arg): self.t})
            new_extension = True
    return new_extension