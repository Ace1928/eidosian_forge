from itertools import product
import math
from .printing import number_to_scientific_html
from ._util import get_backend, mat_dot_vec, prodpow
class NumSysLinTanh(NumSysLin):

    def pre_processor(self, x, params):
        ymax = self.eqsys.upper_conc_bounds(params[:self.eqsys.ns])
        return (np.arctanh((8 * x / ymax - 4) / 5), params)

    def post_processor(self, x, params):
        ymax = self.eqsys.upper_conc_bounds(params[:self.eqsys.ns])
        return (ymax * (4 + 5 * np.tanh(x)) / 8, params)

    def internal_x0_cb(self, init_concs, params):
        return self.pre_processor(init_concs, init_concs)[0]

    def f(self, yvec, params):
        import sympy
        ymax = self.eqsys.upper_conc_bounds(params[:self.eqsys.ns], min_=lambda a, b: sympy.Piecewise((a, a < b), (b, True)))
        ytanh = [yimax * (4 + 5 * sympy.tanh(yi)) / 8 for yimax, yi in zip(ymax, yvec)]
        return NumSysLin.f(self, ytanh, params)