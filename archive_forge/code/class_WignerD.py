from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.simplify import simplify
from sympy.matrices import zeros
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import pretty_symbol
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.operator import (HermitianOperator, Operator,
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import ComplexSpace, DirectSumHilbertSpace
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.qapply import qapply
class WignerD(Expr):
    """Wigner-D function

    The Wigner D-function gives the matrix elements of the rotation
    operator in the jm-representation. For the Euler angles `\\alpha`,
    `\\beta`, `\\gamma`, the D-function is defined such that:

    .. math ::
        <j,m| \\mathcal{R}(\\alpha, \\beta, \\gamma ) |j',m'> = \\delta_{jj'} D(j, m, m', \\alpha, \\beta, \\gamma)

    Where the rotation operator is as defined by the Rotation class [1]_.

    The Wigner D-function defined in this way gives:

    .. math ::
        D(j, m, m', \\alpha, \\beta, \\gamma) = e^{-i m \\alpha} d(j, m, m', \\beta) e^{-i m' \\gamma}

    Where d is the Wigner small-d function, which is given by Rotation.d.

    The Wigner small-d function gives the component of the Wigner
    D-function that is determined by the second Euler angle. That is the
    Wigner D-function is:

    .. math ::
        D(j, m, m', \\alpha, \\beta, \\gamma) = e^{-i m \\alpha} d(j, m, m', \\beta) e^{-i m' \\gamma}

    Where d is the small-d function. The Wigner D-function is given by
    Rotation.D.

    Note that to evaluate the D-function, the j, m and mp parameters must
    be integer or half integer numbers.

    Parameters
    ==========

    j : Number
        Total angular momentum
    m : Number
        Eigenvalue of angular momentum along axis after rotation
    mp : Number
        Eigenvalue of angular momentum along rotated axis
    alpha : Number, Symbol
        First Euler angle of rotation
    beta : Number, Symbol
        Second Euler angle of rotation
    gamma : Number, Symbol
        Third Euler angle of rotation

    Examples
    ========

    Evaluate the Wigner-D matrix elements of a simple rotation:

        >>> from sympy.physics.quantum.spin import Rotation
        >>> from sympy import pi
        >>> rot = Rotation.D(1, 1, 0, pi, pi/2, 0)
        >>> rot
        WignerD(1, 1, 0, pi, pi/2, 0)
        >>> rot.doit()
        sqrt(2)/2

    Evaluate the Wigner-d matrix elements of a simple rotation

        >>> rot = Rotation.d(1, 1, 0, pi/2)
        >>> rot
        WignerD(1, 1, 0, 0, pi/2, 0)
        >>> rot.doit()
        -sqrt(2)/2

    See Also
    ========

    Rotation: Rotation operator

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """
    is_commutative = True

    def __new__(cls, *args, **hints):
        if not len(args) == 6:
            raise ValueError('6 parameters expected, got %s' % args)
        args = sympify(args)
        evaluate = hints.get('evaluate', False)
        if evaluate:
            return Expr.__new__(cls, *args)._eval_wignerd()
        return Expr.__new__(cls, *args)

    @property
    def j(self):
        return self.args[0]

    @property
    def m(self):
        return self.args[1]

    @property
    def mp(self):
        return self.args[2]

    @property
    def alpha(self):
        return self.args[3]

    @property
    def beta(self):
        return self.args[4]

    @property
    def gamma(self):
        return self.args[5]

    def _latex(self, printer, *args):
        if self.alpha == 0 and self.gamma == 0:
            return 'd^{%s}_{%s,%s}\\left(%s\\right)' % (printer._print(self.j), printer._print(self.m), printer._print(self.mp), printer._print(self.beta))
        return 'D^{%s}_{%s,%s}\\left(%s,%s,%s\\right)' % (printer._print(self.j), printer._print(self.m), printer._print(self.mp), printer._print(self.alpha), printer._print(self.beta), printer._print(self.gamma))

    def _pretty(self, printer, *args):
        top = printer._print(self.j)
        bot = printer._print(self.m)
        bot = prettyForm(*bot.right(','))
        bot = prettyForm(*bot.right(printer._print(self.mp)))
        pad = max(top.width(), bot.width())
        top = prettyForm(*top.left(' '))
        bot = prettyForm(*bot.left(' '))
        if pad > top.width():
            top = prettyForm(*top.right(' ' * (pad - top.width())))
        if pad > bot.width():
            bot = prettyForm(*bot.right(' ' * (pad - bot.width())))
        if self.alpha == 0 and self.gamma == 0:
            args = printer._print(self.beta)
            s = stringPict('d' + ' ' * pad)
        else:
            args = printer._print(self.alpha)
            args = prettyForm(*args.right(','))
            args = prettyForm(*args.right(printer._print(self.beta)))
            args = prettyForm(*args.right(','))
            args = prettyForm(*args.right(printer._print(self.gamma)))
            s = stringPict('D' + ' ' * pad)
        args = prettyForm(*args.parens())
        s = prettyForm(*s.above(top))
        s = prettyForm(*s.below(bot))
        s = prettyForm(*s.right(args))
        return s

    def doit(self, **hints):
        hints['evaluate'] = True
        return WignerD(*self.args, **hints)

    def _eval_wignerd(self):
        j = self.j
        m = self.m
        mp = self.mp
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        if alpha == 0 and beta == 0 and (gamma == 0):
            return KroneckerDelta(m, mp)
        if not j.is_number:
            raise ValueError('j parameter must be numerical to evaluate, got %s' % j)
        r = 0
        if beta == pi / 2:
            for k in range(2 * j + 1):
                if k > j + mp or k > j - m or k < mp - m:
                    continue
                r += S.NegativeOne ** k * binomial(j + mp, k) * binomial(j - mp, k + m - mp)
            r *= S.NegativeOne ** (m - mp) / 2 ** j * sqrt(factorial(j + m) * factorial(j - m) / (factorial(j + mp) * factorial(j - mp)))
        else:
            size, mvals = m_values(j)
            for mpp in mvals:
                r += Rotation.d(j, m, mpp, pi / 2).doit() * (cos(-mpp * beta) + I * sin(-mpp * beta)) * Rotation.d(j, mpp, -mp, pi / 2).doit()
            r = r * I ** (2 * j - m - mp) * (-1) ** (2 * m)
            r = simplify(r)
        r *= exp(-I * m * alpha) * exp(-I * mp * gamma)
        return r