from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j, wigner_9j
from sympy.printing.precedence import PRECEDENCE
class CG(Wigner3j):
    """Class for Clebsch-Gordan coefficient.

    Explanation
    ===========

    Clebsch-Gordan coefficients describe the angular momentum coupling between
    two systems. The coefficients give the expansion of a coupled total angular
    momentum state and an uncoupled tensor product state. The Clebsch-Gordan
    coefficients are defined as [1]_:

    .. math ::
        C^{j_3,m_3}_{j_1,m_1,j_2,m_2} = \\left\\langle j_1,m_1;j_2,m_2 | j_3,m_3\\right\\rangle

    Parameters
    ==========

    j1, m1, j2, m2 : Number, Symbol
        Angular momenta of states 1 and 2.

    j3, m3: Number, Symbol
        Total angular momentum of the coupled system.

    Examples
    ========

    Define a Clebsch-Gordan coefficient and evaluate its value

        >>> from sympy.physics.quantum.cg import CG
        >>> from sympy import S
        >>> cg = CG(S(3)/2, S(3)/2, S(1)/2, -S(1)/2, 1, 1)
        >>> cg
        CG(3/2, 3/2, 1/2, -1/2, 1, 1)
        >>> cg.doit()
        sqrt(3)/2
        >>> CG(j1=S(1)/2, m1=-S(1)/2, j2=S(1)/2, m2=+S(1)/2, j3=1, m3=0).doit()
        sqrt(2)/2


    Compare [2]_.

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    .. [2] `Clebsch-Gordan Coefficients, Spherical Harmonics, and d Functions
        <https://pdg.lbl.gov/2020/reviews/rpp2020-rev-clebsch-gordan-coefs.pdf>`_
        in P.A. Zyla *et al.* (Particle Data Group), Prog. Theor. Exp. Phys.
        2020, 083C01 (2020).
    """
    precedence = PRECEDENCE['Pow'] - 1

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError('Coefficients must be numerical')
        return clebsch_gordan(self.j1, self.j2, self.j3, self.m1, self.m2, self.m3)

    def _pretty(self, printer, *args):
        bot = printer._print_seq((self.j1, self.m1, self.j2, self.m2), delimiter=',')
        top = printer._print_seq((self.j3, self.m3), delimiter=',')
        pad = max(top.width(), bot.width())
        bot = prettyForm(*bot.left(' '))
        top = prettyForm(*top.left(' '))
        if not pad == bot.width():
            bot = prettyForm(*bot.right(' ' * (pad - bot.width())))
        if not pad == top.width():
            top = prettyForm(*top.right(' ' * (pad - top.width())))
        s = stringPict('C' + ' ' * pad)
        s = prettyForm(*s.below(bot))
        s = prettyForm(*s.above(top))
        return s

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j3, self.m3, self.j1, self.m1, self.j2, self.m2))
        return 'C^{%s,%s}_{%s,%s,%s,%s}' % tuple(label)