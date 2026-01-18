from sympy.utilities.lambdify import lambdify
from sympy.core.numbers import pi
from sympy.functions import sin, cos
from sympy.plotting.pygletplot.plot_curve import PlotCurve
from sympy.plotting.pygletplot.plot_surface import PlotSurface
from math import sin as p_sin
from math import cos as p_cos
class Spherical(PlotSurface):
    i_vars, d_vars = ('tp', 'r')
    intervals = [[0, 2 * pi, 40], [0, pi, 20]]
    aliases = ['spherical']
    is_default = False

    def _get_sympy_evaluator(self):
        fr = self.d_vars[0]
        t = self.u_interval.v
        p = self.v_interval.v

        def e(_t, _p):
            _r = float(fr.subs(t, _t).subs(p, _p))
            return (_r * p_cos(_t) * p_sin(_p), _r * p_sin(_t) * p_sin(_p), _r * p_cos(_p))
        return e

    def _get_lambda_evaluator(self):
        fr = self.d_vars[0]
        t = self.u_interval.v
        p = self.v_interval.v
        fx = fr * cos(t) * sin(p)
        fy = fr * sin(t) * sin(p)
        fz = fr * cos(p)
        return lambdify([t, p], [fx, fy, fz])