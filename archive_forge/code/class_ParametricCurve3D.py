from sympy.utilities.lambdify import lambdify
from sympy.core.numbers import pi
from sympy.functions import sin, cos
from sympy.plotting.pygletplot.plot_curve import PlotCurve
from sympy.plotting.pygletplot.plot_surface import PlotSurface
from math import sin as p_sin
from math import cos as p_cos
class ParametricCurve3D(PlotCurve):
    i_vars, d_vars = ('t', 'xyz')
    intervals = [[0, 2 * pi, 100]]
    aliases = ['parametric']
    is_default = True

    def _get_sympy_evaluator(self):
        fx, fy, fz = self.d_vars
        t = self.t_interval.v

        @float_vec3
        def e(_t):
            return (fx.subs(t, _t), fy.subs(t, _t), fz.subs(t, _t))
        return e

    def _get_lambda_evaluator(self):
        fx, fy, fz = self.d_vars
        t = self.t_interval.v
        return lambdify([t], [fx, fy, fz])