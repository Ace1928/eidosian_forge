from sympy.core import S, Function, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name
from itertools import product
def _eval_integral(self, x, _first=True, **kwargs):
    """Return the indefinite integral of the
        Piecewise such that subsequent substitution of x with a
        value will give the value of the integral (not including
        the constant of integration) up to that point. To only
        integrate the individual parts of Piecewise, use the
        ``piecewise_integrate`` method.

        Examples
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> p = Piecewise((0, x < 0), (1, x < 1), (2, True))
        >>> p.integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x - 1, True))
        >>> p.piecewise_integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x, True))

        See Also
        ========
        Piecewise.piecewise_integrate
        """
    from sympy.integrals.integrals import integrate
    if _first:

        def handler(ipw):
            if isinstance(ipw, self.func):
                return ipw._eval_integral(x, _first=False, **kwargs)
            else:
                return ipw.integrate(x, **kwargs)
        irv = self._handle_irel(x, handler)
        if irv is not None:
            return irv
    ok, abei = self._intervals(x)
    if not ok:
        from sympy.integrals.integrals import Integral
        return Integral(self, x)
    pieces = [(a, b) for a, b, _, _ in abei]
    oo = S.Infinity
    done = [(-oo, oo, -1)]
    for k, p in enumerate(pieces):
        if p == (-oo, oo):
            for j, (a, b, i) in enumerate(done):
                if i == -1:
                    done[j] = (a, b, k)
            break
        N = len(done) - 1
        for j, (a, b, i) in enumerate(reversed(done)):
            if i == -1:
                j = N - j
                done[j:j + 1] = _clip(p, (a, b), k)
    done = [(a, b, i) for a, b, i in done if a != b]
    if any((i == -1 for a, b, i in done)):
        abei.append((-oo, oo, Undefined, -1))
    args = []
    sum = None
    for a, b, i in done:
        anti = integrate(abei[i][-2], x, **kwargs)
        if sum is None:
            sum = anti
        else:
            sum = sum.subs(x, a)
            e = anti._eval_interval(x, a, x)
            if sum.has(*_illegal) or e.has(*_illegal):
                sum = anti
            else:
                sum += e
        if b is S.Infinity:
            cond = True
        elif self.args[abei[i][-1]].cond.subs(x, b) == False:
            cond = x < b
        else:
            cond = x <= b
        args.append((sum, cond))
    return Piecewise(*args)