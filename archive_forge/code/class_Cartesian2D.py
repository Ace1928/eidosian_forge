from sympy.utilities.lambdify import lambdify
from sympy.core.numbers import pi
from sympy.functions import sin, cos
from sympy.plotting.pygletplot.plot_curve import PlotCurve
from sympy.plotting.pygletplot.plot_surface import PlotSurface
from math import sin as p_sin
from math import cos as p_cos
class Cartesian2D(PlotCurve):
    i_vars, d_vars = ('x', 'y')
    intervals = [[-5, 5, 100]]
    aliases = ['cartesian']
    is_default = True

    def _get_sympy_evaluator(self):
        fy = self.d_vars[0]
        x = self.t_interval.v

        @float_vec3
        def e(_x):
            return (_x, fy.subs(x, _x), 0.0)
        return e

    def _get_lambda_evaluator(self):
        fy = self.d_vars[0]
        x = self.t_interval.v
        return lambdify([x], [x, fy, 0.0])