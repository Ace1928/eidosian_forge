from sympy.utilities.lambdify import lambdify
from sympy.core.numbers import pi
from sympy.functions import sin, cos
from sympy.plotting.pygletplot.plot_curve import PlotCurve
from sympy.plotting.pygletplot.plot_surface import PlotSurface
from math import sin as p_sin
from math import cos as p_cos
class Cartesian3D(PlotSurface):
    i_vars, d_vars = ('xy', 'z')
    intervals = [[-1, 1, 40], [-1, 1, 40]]
    aliases = ['cartesian', 'monge']
    is_default = True

    def _get_sympy_evaluator(self):
        fz = self.d_vars[0]
        x = self.u_interval.v
        y = self.v_interval.v

        @float_vec3
        def e(_x, _y):
            return (_x, _y, fz.subs(x, _x).subs(y, _y))
        return e

    def _get_lambda_evaluator(self):
        fz = self.d_vars[0]
        x = self.u_interval.v
        y = self.v_interval.v
        return lambdify([x, y], [x, y, fz])