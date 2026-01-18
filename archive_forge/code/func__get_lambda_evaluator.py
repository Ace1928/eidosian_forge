from sympy.utilities.lambdify import lambdify
from sympy.core.numbers import pi
from sympy.functions import sin, cos
from sympy.plotting.pygletplot.plot_curve import PlotCurve
from sympy.plotting.pygletplot.plot_surface import PlotSurface
from math import sin as p_sin
from math import cos as p_cos
def _get_lambda_evaluator(self):
    fr = self.d_vars[0]
    t = self.u_interval.v
    p = self.v_interval.v
    fx = fr * cos(t) * sin(p)
    fy = fr * sin(t) * sin(p)
    fz = fr * cos(p)
    return lambdify([t, p], [fx, fy, fz])