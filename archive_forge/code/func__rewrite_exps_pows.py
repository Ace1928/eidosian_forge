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
def _rewrite_exps_pows(self, exps, pows, numpows, sympows, log_new_extension):
    """
        Rewrite exps/pows for better processing.
        """
    from .prde import is_deriv_k
    ratpows = [i for i in self.newf.atoms(Pow) if isinstance(i.base, exp) and i.exp.is_Rational]
    ratpows_repl = [(i, i.base.base ** (i.exp * i.base.exp)) for i in ratpows]
    self.backsubs += [(j, i) for i, j in ratpows_repl]
    self.newf = self.newf.xreplace(dict(ratpows_repl))
    exps = update_sets(exps, self.newf.atoms(exp), lambda i: i.exp.is_rational_function(*self.T) and i.exp.has(*self.T))
    pows = update_sets(pows, self.newf.atoms(Pow), lambda i: i.exp.is_rational_function(*self.T) and i.exp.has(*self.T))
    numpows = update_sets(numpows, set(pows), lambda i: not i.base.has(*self.T))
    sympows = update_sets(sympows, set(pows) - set(numpows), lambda i: i.base.is_rational_function(*self.T) and (not i.exp.is_Integer))
    for i in ordered(pows):
        old = i
        new = exp(i.exp * log(i.base))
        if i in sympows:
            if i.exp.is_Rational:
                raise NotImplementedError('Algebraic extensions are not supported (%s).' % str(i))
            basea, based = frac_in(i.base, self.t)
            A = is_deriv_k(basea, based, self)
            if A is None:
                self.newf = self.newf.xreplace({old: new})
                self.backsubs += [(new, old)]
                log_new_extension = self._log_part([log(i.base)])
                exps = update_sets(exps, self.newf.atoms(exp), lambda i: i.exp.is_rational_function(*self.T) and i.exp.has(*self.T))
                continue
            ans, u, const = A
            newterm = exp(i.exp * (log(const) + u))
            self.newf = self.newf.xreplace({i: newterm})
        elif i not in numpows:
            continue
        else:
            newterm = new
        self.backsubs.append((new, old))
        self.newf = self.newf.xreplace({old: newterm})
        exps.append(newterm)
    return (exps, pows, numpows, sympows, log_new_extension)