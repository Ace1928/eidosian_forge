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
def _exp_part(self, exps):
    """
        Try to build an exponential extension.

        Returns
        =======

        Returns True if there was a new extension, False if there was no new
        extension but it was able to rewrite the given exponentials in terms
        of the existing extension, and None if the entire extension building
        process should be restarted.  If the process fails because there is no
        way around an algebraic extension (e.g., exp(log(x)/2)), it will raise
        NotImplementedError.
        """
    from .prde import is_log_deriv_k_t_radical
    new_extension = False
    restart = False
    expargs = [i.exp for i in exps]
    ip = integer_powers(expargs)
    for arg, others in ip:
        others.sort(key=lambda i: i[1])
        arga, argd = frac_in(arg, self.t)
        A = is_log_deriv_k_t_radical(arga, argd, self)
        if A is not None:
            ans, u, n, const = A
            if n == -1:
                n = 1
                u **= -1
                const *= -1
                ans = [(i, -j) for i, j in ans]
            if n == 1:
                self.newf = self.newf.xreplace({exp(arg): exp(const) * Mul(*[u ** power for u, power in ans])})
                self.newf = self.newf.xreplace({exp(p * exparg): exp(const * p) * Mul(*[u ** power for u, power in ans]) for exparg, p in others})
                continue
            elif const or len(ans) > 1:
                rad = Mul(*[term ** (power / n) for term, power in ans])
                self.newf = self.newf.xreplace({exp(p * exparg): exp(const * p) * rad for exparg, p in others})
                self.newf = self.newf.xreplace(dict(list(zip(reversed(self.T), reversed([f(self.x) for f in self.Tfuncs])))))
                restart = True
                break
            else:
                raise NotImplementedError('Cannot integrate over algebraic extensions.')
        else:
            arga, argd = frac_in(arg, self.t)
            darga = argd * derivation(Poly(arga, self.t), self) - arga * derivation(Poly(argd, self.t), self)
            dargd = argd ** 2
            darga, dargd = darga.cancel(dargd, include=True)
            darg = darga.as_expr() / dargd.as_expr()
            self.t = next(self.ts)
            self.T.append(self.t)
            self.extargs.append(arg)
            self.exts.append('exp')
            self.D.append(darg.as_poly(self.t, expand=False) * Poly(self.t, self.t, expand=False))
            if self.dummy:
                i = Dummy('i')
            else:
                i = Symbol('i')
            self.Tfuncs += [Lambda(i, exp(arg.subs(self.x, i)))]
            self.newf = self.newf.xreplace({exp(exparg): self.t ** p for exparg, p in others})
            new_extension = True
    if restart:
        return None
    return new_extension