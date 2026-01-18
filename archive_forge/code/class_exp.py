from itertools import product
from typing import Tuple as tTuple
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (Function, ArgumentIndexError, expand_log,
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, pi, I, ImaginaryUnit
from sympy.core.parameters import global_parameters
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory import multiplicity, perfect_power
from sympy.ntheory.factor_ import factorint
class exp(ExpBase, metaclass=ExpMeta):
    """
    The exponential function, :math:`e^x`.

    Examples
    ========

    >>> from sympy import exp, I, pi
    >>> from sympy.abc import x
    >>> exp(x)
    exp(x)
    >>> exp(x).diff(x)
    exp(x)
    >>> exp(I*pi)
    -1

    Parameters
    ==========

    arg : Expr

    See Also
    ========

    log
    """

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return self
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_refine(self, assumptions):
        from sympy.assumptions import ask, Q
        arg = self.args[0]
        if arg.is_Mul:
            Ioo = I * S.Infinity
            if arg in [Ioo, -Ioo]:
                return S.NaN
            coeff = arg.as_coefficient(pi * I)
            if coeff:
                if ask(Q.integer(2 * coeff)):
                    if ask(Q.even(coeff)):
                        return S.One
                    elif ask(Q.odd(coeff)):
                        return S.NegativeOne
                    elif ask(Q.even(coeff + S.Half)):
                        return -I
                    elif ask(Q.odd(coeff + S.Half)):
                        return I

    @classmethod
    def eval(cls, arg):
        from sympy.calculus import AccumBounds
        from sympy.matrices.matrices import MatrixBase
        from sympy.sets.setexpr import SetExpr
        from sympy.simplify.simplify import logcombine
        if isinstance(arg, MatrixBase):
            return arg.exp()
        elif global_parameters.exp_is_pow:
            return Pow(S.Exp1, arg)
        elif arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.One
            elif arg is S.One:
                return S.Exp1
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
        elif arg is S.ComplexInfinity:
            return S.NaN
        elif isinstance(arg, log):
            return arg.args[0]
        elif isinstance(arg, AccumBounds):
            return AccumBounds(exp(arg.min), exp(arg.max))
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)
        elif arg.is_Mul:
            coeff = arg.as_coefficient(pi * I)
            if coeff:
                if (2 * coeff).is_integer:
                    if coeff.is_even:
                        return S.One
                    elif coeff.is_odd:
                        return S.NegativeOne
                    elif (coeff + S.Half).is_even:
                        return -I
                    elif (coeff + S.Half).is_odd:
                        return I
                elif coeff.is_Rational:
                    ncoeff = coeff % 2
                    if ncoeff > 1:
                        ncoeff -= 2
                    if ncoeff != coeff:
                        return cls(ncoeff * pi * I)
            coeff, terms = arg.as_coeff_Mul()
            if coeff in [S.NegativeInfinity, S.Infinity]:
                if terms.is_number:
                    if coeff is S.NegativeInfinity:
                        terms = -terms
                    if re(terms).is_zero and terms is not S.Zero:
                        return S.NaN
                    if re(terms).is_positive and im(terms) is not S.Zero:
                        return S.ComplexInfinity
                    if re(terms).is_negative:
                        return S.Zero
                return None
            coeffs, log_term = ([coeff], None)
            for term in Mul.make_args(terms):
                term_ = logcombine(term)
                if isinstance(term_, log):
                    if log_term is None:
                        log_term = term_.args[0]
                    else:
                        return None
                elif term.is_comparable:
                    coeffs.append(term)
                else:
                    return None
            return log_term ** Mul(*coeffs) if log_term else None
        elif arg.is_Add:
            out = []
            add = []
            argchanged = False
            for a in arg.args:
                if a is S.One:
                    add.append(a)
                    continue
                newa = cls(a)
                if isinstance(newa, cls):
                    if newa.args[0] != a:
                        add.append(newa.args[0])
                        argchanged = True
                    else:
                        add.append(a)
                else:
                    out.append(newa)
            if out or argchanged:
                return Mul(*out) * cls(Add(*add), evaluate=False)
        if arg.is_zero:
            return S.One

    @property
    def base(self):
        """
        Returns the base of the exponential function.
        """
        return S.Exp1

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Calculates the next term in the Taylor series expansion.
        """
        if n < 0:
            return S.Zero
        if n == 0:
            return S.One
        x = sympify(x)
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return p * x / n
        return x ** n / factorial(n)

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a 2-tuple representing a complex number.

        Examples
        ========

        >>> from sympy import exp, I
        >>> from sympy.abc import x
        >>> exp(x).as_real_imag()
        (exp(re(x))*cos(im(x)), exp(re(x))*sin(im(x)))
        >>> exp(1).as_real_imag()
        (E, 0)
        >>> exp(I).as_real_imag()
        (cos(1), sin(1))
        >>> exp(1+I).as_real_imag()
        (E*cos(1), E*sin(1))

        See Also
        ========

        sympy.functions.elementary.complexes.re
        sympy.functions.elementary.complexes.im
        """
        from sympy.functions.elementary.trigonometric import cos, sin
        re, im = self.args[0].as_real_imag()
        if deep:
            re = re.expand(deep, **hints)
            im = im.expand(deep, **hints)
        cos, sin = (cos(im), sin(im))
        return (exp(re) * cos, exp(re) * sin)

    def _eval_subs(self, old, new):
        if old.is_Pow:
            old = exp(old.exp * log(old.base))
        elif old is S.Exp1 and new.is_Function:
            old = exp
        if isinstance(old, exp) or old is S.Exp1:
            f = lambda a: Pow(*a.as_base_exp(), evaluate=False) if a.is_Pow or isinstance(a, exp) else a
            return Pow._eval_subs(f(self), f(old), new)
        if old is exp and (not new.is_Function):
            return new ** self.exp._subs(old, new)
        return Function._eval_subs(self, old, new)

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True
        elif self.args[0].is_imaginary:
            arg2 = -S(2) * I * self.args[0] / pi
            return arg2.is_even

    def _eval_is_complex(self):

        def complex_extended_negative(arg):
            yield arg.is_complex
            yield arg.is_extended_negative
        return fuzzy_or(complex_extended_negative(self.args[0]))

    def _eval_is_algebraic(self):
        if (self.exp / pi / I).is_rational:
            return True
        if fuzzy_not(self.exp.is_zero):
            if self.exp.is_algebraic:
                return False
            elif (self.exp / pi).is_rational:
                return False

    def _eval_is_extended_positive(self):
        if self.exp.is_extended_real:
            return self.args[0] is not S.NegativeInfinity
        elif self.exp.is_imaginary:
            arg2 = -I * self.args[0] / pi
            return arg2.is_even

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.functions.elementary.complexes import sign
        from sympy.functions.elementary.integers import ceiling
        from sympy.series.limits import limit
        from sympy.series.order import Order
        from sympy.simplify.powsimp import powsimp
        arg = self.exp
        arg_series = arg._eval_nseries(x, n=n, logx=logx)
        if arg_series.is_Order:
            return 1 + arg_series
        arg0 = limit(arg_series.removeO(), x, 0)
        if arg0 is S.NegativeInfinity:
            return Order(x ** n, x)
        if arg0 is S.Infinity:
            return self
        if any((isinstance(arg, (sign, ImaginaryUnit)) for arg in arg0.args)):
            return self
        t = Dummy('t')
        nterms = n
        try:
            cf = Order(arg.as_leading_term(x, logx=logx), x).getn()
        except (NotImplementedError, PoleError):
            cf = 0
        if cf and cf > 0:
            nterms = ceiling(n / cf)
        exp_series = exp(t)._taylor(t, nterms)
        r = exp(arg0) * exp_series.subs(t, arg_series - arg0)
        rep = {logx: log(x)} if logx is not None else {}
        if r.subs(rep) == self:
            return r
        if cf and cf > 1:
            r += Order((arg_series - arg0) ** n, x) / x ** ((cf - 1) * n)
        else:
            r += Order((arg_series - arg0) ** n, x)
        r = r.expand()
        r = powsimp(r, deep=True, combine='exp')
        simplerat = lambda x: x.is_Rational and x.q in [3, 4, 6]
        w = Wild('w', properties=[simplerat])
        r = r.replace(S.NegativeOne ** w, expand_complex(S.NegativeOne ** w))
        return r

    def _taylor(self, x, n):
        l = []
        g = None
        for i in range(n):
            g = self.taylor_term(i, self.args[0], g)
            g = g.nseries(x, n=n)
            l.append(g.removeO())
        return Add(*l)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.calculus.util import AccumBounds
        arg = self.args[0].cancel().as_leading_term(x, logx=logx)
        arg0 = arg.subs(x, 0)
        if arg is S.NaN:
            return S.NaN
        if isinstance(arg0, AccumBounds):
            if re(cdir) < S.Zero:
                return exp(-arg0)
            return exp(arg0)
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0)
        if arg0.is_infinite is False:
            return exp(arg0)
        raise PoleError('Cannot expand %s around 0' % self)

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import sin
        return sin(I * arg + pi / 2) - I * sin(I * arg)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import cos
        return cos(I * arg) + I * cos(I * arg + pi / 2)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        from sympy.functions.elementary.hyperbolic import tanh
        return (1 + tanh(arg / 2)) / (1 - tanh(arg / 2))

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import sin, cos
        if arg.is_Mul:
            coeff = arg.coeff(pi * I)
            if coeff and coeff.is_number:
                cosine, sine = (cos(pi * coeff), sin(pi * coeff))
                if not isinstance(cosine, cos) and (not isinstance(sine, sin)):
                    return cosine + I * sine

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if arg.is_Mul:
            logs = [a for a in arg.args if isinstance(a, log) and len(a.args) == 1]
            if logs:
                return Pow(logs[0].args[0], arg.coeff(logs[0]))