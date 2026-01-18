from .plot import BaseSeries, Plot
from .experimental_lambdify import experimental_lambdify, vectorized_lambdify
from .intervalmath import interval
from sympy.core.relational import (Equality, GreaterThan, LessThan,
from sympy.core.containers import Tuple
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.logic.boolalg import BooleanFunction
from sympy.polys.polyutils import _sort_gens
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import flatten
import warnings
def _get_raster_interval(self, func):
    """ Uses interval math to adaptively mesh and obtain the plot"""
    k = self.depth
    interval_list = []
    np = import_module('numpy')
    xsample = np.linspace(self.start_x, self.end_x, 33)
    ysample = np.linspace(self.start_y, self.end_y, 33)
    jitterx = (np.random.rand(len(xsample)) * 2 - 1) * (self.end_x - self.start_x) / 2 ** 20
    jittery = (np.random.rand(len(ysample)) * 2 - 1) * (self.end_y - self.start_y) / 2 ** 20
    xsample += jitterx
    ysample += jittery
    xinter = [interval(x1, x2) for x1, x2 in zip(xsample[:-1], xsample[1:])]
    yinter = [interval(y1, y2) for y1, y2 in zip(ysample[:-1], ysample[1:])]
    interval_list = [[x, y] for x in xinter for y in yinter]
    plot_list = []

    def refine_pixels(interval_list):
        """ Evaluates the intervals and subdivides the interval if the
            expression is partially satisfied."""
        temp_interval_list = []
        plot_list = []
        for intervals in interval_list:
            intervalx = intervals[0]
            intervaly = intervals[1]
            func_eval = func(intervalx, intervaly)
            if func_eval[1] is False or func_eval[0] is False:
                pass
            elif func_eval == (True, True):
                plot_list.append([intervalx, intervaly])
            elif func_eval[1] is None or func_eval[0] is None:
                avgx = intervalx.mid
                avgy = intervaly.mid
                a = interval(intervalx.start, avgx)
                b = interval(avgx, intervalx.end)
                c = interval(intervaly.start, avgy)
                d = interval(avgy, intervaly.end)
                temp_interval_list.append([a, c])
                temp_interval_list.append([a, d])
                temp_interval_list.append([b, c])
                temp_interval_list.append([b, d])
        return (temp_interval_list, plot_list)
    while k >= 0 and len(interval_list):
        interval_list, plot_list_temp = refine_pixels(interval_list)
        plot_list.extend(plot_list_temp)
        k = k - 1
    if self.has_equality:
        for intervals in interval_list:
            intervalx = intervals[0]
            intervaly = intervals[1]
            func_eval = func(intervalx, intervaly)
            if func_eval[1] and func_eval[0] is not False:
                plot_list.append([intervalx, intervaly])
    return (plot_list, 'fill')