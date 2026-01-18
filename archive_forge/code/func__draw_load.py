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
def _draw_load(self, pictorial, length, l):
    loads = list(set(self.applied_loads) - set(self._support_as_loads))
    height = length / 10
    x = self.variable
    annotations = []
    markers = []
    load_args = []
    scaled_load = 0
    load_args1 = []
    scaled_load1 = 0
    load_eq = 0
    load_eq1 = 0
    fill = None
    plus = 0
    minus = 0
    for load in loads:
        if l:
            pos = load[1].subs(l)
        else:
            pos = load[1]
        if load[2] == -1:
            if isinstance(load[0], Symbol) or load[0].is_negative:
                annotations.append({'text': '', 'xy': (pos, 0), 'xytext': (pos, height - 4 * height), 'arrowprops': {'width': 1.5, 'headlength': 5, 'headwidth': 5, 'facecolor': 'black'}})
            else:
                annotations.append({'text': '', 'xy': (pos, height), 'xytext': (pos, height * 4), 'arrowprops': {'width': 1.5, 'headlength': 4, 'headwidth': 4, 'facecolor': 'black'}})
        elif load[2] == -2:
            if load[0].is_negative:
                markers.append({'args': [[pos], [height / 2]], 'marker': '$\\circlearrowright$', 'markersize': 15})
            else:
                markers.append({'args': [[pos], [height / 2]], 'marker': '$\\circlearrowleft$', 'markersize': 15})
        elif load[2] >= 0:
            value, start, order, end = load
            if value > 0:
                plus = 1
                if pictorial:
                    value = 10 ** (1 - order) if order > 0 else length / 2
                    scaled_load += value * SingularityFunction(x, start, order)
                    if end:
                        f2 = 10 ** (1 - order) * x ** order if order > 0 else length / 2 * x ** order
                        for i in range(0, order + 1):
                            scaled_load -= f2.diff(x, i).subs(x, end - start) * SingularityFunction(x, end, i) / factorial(i)
                if pictorial:
                    if isinstance(scaled_load, Add):
                        load_args = scaled_load.args
                    else:
                        load_args = (scaled_load,)
                    load_eq = [i.subs(l) for i in load_args]
                else:
                    if isinstance(self.load, Add):
                        load_args = self.load.args
                    else:
                        load_args = (self.load,)
                    load_eq = [i.subs(l) for i in load_args if list(i.atoms(SingularityFunction))[0].args[2] >= 0]
                load_eq = Add(*load_eq)
                expr = height + load_eq.rewrite(Piecewise)
                y1 = lambdify(x, expr, 'numpy')
            else:
                minus = 1
                if pictorial:
                    value = 10 ** (1 - order) if order > 0 else length / 2
                    scaled_load1 += value * SingularityFunction(x, start, order)
                    if end:
                        f2 = 10 ** (1 - order) * x ** order if order > 0 else length / 2 * x ** order
                        for i in range(0, order + 1):
                            scaled_load1 -= f2.diff(x, i).subs(x, end - start) * SingularityFunction(x, end, i) / factorial(i)
                if pictorial:
                    if isinstance(scaled_load1, Add):
                        load_args1 = scaled_load1.args
                    else:
                        load_args1 = (scaled_load1,)
                    load_eq1 = [i.subs(l) for i in load_args1]
                else:
                    if isinstance(self.load, Add):
                        load_args1 = self.load.args1
                    else:
                        load_args1 = (self.load,)
                    load_eq1 = [i.subs(l) for i in load_args if list(i.atoms(SingularityFunction))[0].args[2] >= 0]
                load_eq1 = -Add(*load_eq1) - height
                expr = height + load_eq1.rewrite(Piecewise)
                y1_ = lambdify(x, expr, 'numpy')
            y = numpy.arange(0, float(length), 0.001)
            y2 = float(height)
            if plus == 1 and minus == 1:
                fill = {'x': y, 'y1': y1(y), 'y2': y1_(y), 'color': 'darkkhaki'}
            elif plus == 1:
                fill = {'x': y, 'y1': y1(y), 'y2': y2, 'color': 'darkkhaki'}
            else:
                fill = {'x': y, 'y1': y1_(y), 'y2': y2, 'color': 'darkkhaki'}
    return (annotations, markers, load_eq, load_eq1, fill)