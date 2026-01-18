from sympy.core import S, diff
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.complexes import im, sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.utilities.misc import filldedent
def _eval_expand_diracdelta(self, **hints):
    """
        Compute a simplified representation of the function using
        property number 4. Pass ``wrt`` as a hint to expand the expression
        with respect to a particular variable.

        Explanation
        ===========

        ``wrt`` is:

        - a variable with respect to which a DiracDelta expression will
        get expanded.

        Examples
        ========

        >>> from sympy import DiracDelta
        >>> from sympy.abc import x, y

        >>> DiracDelta(x*y).expand(diracdelta=True, wrt=x)
        DiracDelta(x)/Abs(y)
        >>> DiracDelta(x*y).expand(diracdelta=True, wrt=y)
        DiracDelta(y)/Abs(x)

        >>> DiracDelta(x**2 + x - 2).expand(diracdelta=True, wrt=x)
        DiracDelta(x - 1)/3 + DiracDelta(x + 2)/3

        See Also
        ========

        is_simple, Diracdelta

        """
    wrt = hints.get('wrt', None)
    if wrt is None:
        free = self.free_symbols
        if len(free) == 1:
            wrt = free.pop()
        else:
            raise TypeError(filldedent("\n            When there is more than 1 free symbol or variable in the expression,\n            the 'wrt' keyword is required as a hint to expand when using the\n            DiracDelta hint."))
    if not self.args[0].has(wrt) or (len(self.args) > 1 and self.args[1] != 0):
        return self
    try:
        argroots = roots(self.args[0], wrt)
        result = 0
        valid = True
        darg = abs(diff(self.args[0], wrt))
        for r, m in argroots.items():
            if r.is_real is not False and m == 1:
                result += self.func(wrt - r) / darg.subs(wrt, r)
            else:
                valid = False
                break
        if valid:
            return result
    except PolynomialError:
        pass
    return self