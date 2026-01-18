from sympy.core import S, Symbol, diff, symbols
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.sympify import sympify
from sympy.solvers import linsolve
from sympy.solvers.ode.ode import dsolve
from sympy.solvers.solvers import solve
from sympy.printing import sstr
from sympy.functions import SingularityFunction, Piecewise, factorial
from sympy.integrals import integrate
from sympy.series import limit
from sympy.plotting import plot, PlotGrid
from sympy.geometry.entity import GeometryEntity
from sympy.external import import_module
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import iterable
def _plot_deflection(self, dir, subs=None):
    deflection = self.deflection()
    if dir == 'x':
        dir_num = 0
        color = 'm'
    elif dir == 'y':
        dir_num = 1
        color = 'r'
    elif dir == 'z':
        dir_num = 2
        color = 'c'
    if subs is None:
        subs = {}
    for sym in deflection[dir_num].atoms(Symbol):
        if sym != self.variable and sym not in subs:
            raise ValueError('Value of %s was not passed.' % sym)
    if self.length in subs:
        length = subs[self.length]
    else:
        length = self.length
    return plot(deflection[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Deflection along %c direction' % dir, xlabel='$\\mathrm{X}$', ylabel='$\\mathrm{\\delta(%c)}$' % dir, line_color=color)