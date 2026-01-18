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
class Wigner6j(Expr):
    """Class for the Wigner-6j symbols

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    """

    def __new__(cls, j1, j2, j12, j3, j, j23):
        args = map(sympify, (j1, j2, j12, j3, j, j23))
        return Expr.__new__(cls, *args)

    @property
    def j1(self):
        return self.args[0]

    @property
    def j2(self):
        return self.args[1]

    @property
    def j12(self):
        return self.args[2]

    @property
    def j3(self):
        return self.args[3]

    @property
    def j(self):
        return self.args[4]

    @property
    def j23(self):
        return self.args[5]

    @property
    def is_symbolic(self):
        return not all((arg.is_number for arg in self.args))

    def _pretty(self, printer, *args):
        m = ((printer._print(self.j1), printer._print(self.j3)), (printer._print(self.j2), printer._print(self.j)), (printer._print(self.j12), printer._print(self.j23)))
        hsep = 2
        vsep = 1
        maxw = [-1] * 3
        for j in range(3):
            maxw[j] = max([m[j][i].width() for i in range(2)])
        D = None
        for i in range(2):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft = wdelta // 2
                wright = wdelta - wleft
                s = prettyForm(*s.right(' ' * wright))
                s = prettyForm(*s.left(' ' * wleft))
                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' ' * hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens(left='{', right='}'))
        return D

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j12, self.j3, self.j, self.j23))
        return '\\left\\{\\begin{array}{ccc} %s & %s & %s \\\\ %s & %s & %s \\end{array}\\right\\}' % tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError('Coefficients must be numerical')
        return wigner_6j(self.j1, self.j2, self.j12, self.j3, self.j, self.j23)